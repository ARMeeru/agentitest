import logging
from typing import Dict, Any, Optional, List, Callable, Union
from enum import Enum
from dataclasses import dataclass

from .base import LLMProviderError, ErrorContext
from .classification import RecoveryStrategy, get_recovery_strategy
from .logging import log_recovery_attempt, get_error_correlation_id


class DegradationLevel(Enum):
    # Levels of service degradation for graceful handling
    FULL_SERVICE = "full_service"           # Normal operation
    REDUCED_FUNCTIONALITY = "reduced"       # Some features disabled
    MINIMAL_SERVICE = "minimal"            # Basic functionality only
    EMERGENCY_MODE = "emergency"           # Critical operations only


@dataclass
class DegradationConfig:
    # Configuration for graceful degradation behavior
    max_retry_attempts: int = 3
    fallback_providers: List[str] = None
    reduced_functionality_threshold: int = 2
    minimal_service_threshold: int = 5
    emergency_mode_threshold: int = 10
    timeout_multiplier: float = 1.5

    def __post_init__(self):
        if self.fallback_providers is None:
            self.fallback_providers = []


class GracefulDegradationManager:
    # Manages graceful degradation for LLM provider failures

    def __init__(self, config: Optional[DegradationConfig] = None):
        self.config = config or DegradationConfig()
        self.provider_failure_counts: Dict[str, int] = {}
        self.current_degradation_level = DegradationLevel.FULL_SERVICE
        self.active_providers: List[str] = []
        self.failed_providers: List[str] = []
        self.logger = logging.getLogger(__name__)

    def register_providers(self, providers: List[str]):
        # Register available LLM providers for fallback
        self.active_providers = providers.copy()
        self.failed_providers = []
        self.provider_failure_counts = {provider: 0 for provider in providers}

    def handle_provider_failure(
        self,
        provider: str,
        exception: Exception,
        context: Optional[ErrorContext] = None
    ) -> Dict[str, Any]:
        # Handle LLM provider failure with graceful degradation
        correlation_id = context.correlation_id if context else get_error_correlation_id()

        # Increment failure count
        self.provider_failure_counts[provider] = self.provider_failure_counts.get(provider, 0) + 1
        failure_count = self.provider_failure_counts[provider]

        self.logger.warning(
            f"Provider {provider} failed (attempt {failure_count}) [correlation_id: {correlation_id}]: {exception}"
        )

        # Determine recovery strategy
        recovery_strategy = get_recovery_strategy(exception, {"retry_count": failure_count})

        # Handle different recovery strategies
        if recovery_strategy == RecoveryStrategy.FALLBACK_PROVIDER:
            return self._handle_provider_fallback(provider, exception, correlation_id)

        elif recovery_strategy == RecoveryStrategy.GRACEFUL_DEGRADATION:
            return self._handle_graceful_degradation(provider, exception, correlation_id)

        elif recovery_strategy == RecoveryStrategy.RETRY_WITH_BACKOFF:
            if failure_count <= self.config.max_retry_attempts:
                return self._handle_retry_with_backoff(provider, failure_count, correlation_id)
            else:
                return self._handle_provider_fallback(provider, exception, correlation_id)

        elif recovery_strategy == RecoveryStrategy.CIRCUIT_BREAKER:
            return self._handle_circuit_breaker(provider, exception, correlation_id)

        else:  # FAIL_FAST
            return self._handle_fail_fast(provider, exception, correlation_id)

    def _handle_provider_fallback(
        self,
        failed_provider: str,
        exception: Exception,
        correlation_id: str
    ) -> Dict[str, Any]:
        # Attempt to fallback to alternative provider

        # Mark provider as failed
        if failed_provider in self.active_providers:
            self.active_providers.remove(failed_provider)
            self.failed_providers.append(failed_provider)

        # Find next available provider
        fallback_provider = self._get_next_available_provider(failed_provider)

        if fallback_provider:
            log_recovery_attempt(
                correlation_id=correlation_id,
                strategy="provider_fallback",
                attempt_number=1,
                success=True,
                failed_provider=failed_provider,
                fallback_provider=fallback_provider
            )

            return {
                "strategy": "provider_fallback",
                "success": True,
                "fallback_provider": fallback_provider,
                "failed_provider": failed_provider,
                "message": f"Switched from {failed_provider} to {fallback_provider}",
                "degradation_level": self._update_degradation_level()
            }
        else:
            # No fallback available - enter degraded mode
            return self._handle_graceful_degradation(failed_provider, exception, correlation_id)

    def _handle_graceful_degradation(
        self,
        provider: str,
        exception: Exception,
        correlation_id: str
    ) -> Dict[str, Any]:
        # Enter graceful degradation mode

        total_failures = sum(self.provider_failure_counts.values())
        new_degradation_level = self._calculate_degradation_level(total_failures)

        if new_degradation_level != self.current_degradation_level:
            self.current_degradation_level = new_degradation_level

            self.logger.warning(
                f"Entering degradation level: {new_degradation_level.value} "
                f"[correlation_id: {correlation_id}]"
            )

        log_recovery_attempt(
            correlation_id=correlation_id,
            strategy="graceful_degradation",
            attempt_number=1,
            success=True,
            degradation_level=new_degradation_level.value,
            total_failures=total_failures
        )

        return {
            "strategy": "graceful_degradation",
            "success": True,
            "degradation_level": new_degradation_level.value,
            "message": self._get_degradation_message(new_degradation_level),
            "available_features": self._get_available_features(new_degradation_level),
            "timeout_multiplier": self._get_timeout_multiplier(new_degradation_level)
        }

    def _handle_retry_with_backoff(
        self,
        provider: str,
        attempt_number: int,
        correlation_id: str
    ) -> Dict[str, Any]:
        # Handle retry with exponential backoff

        backoff_seconds = 2 ** (attempt_number - 1)  # Exponential backoff

        log_recovery_attempt(
            correlation_id=correlation_id,
            strategy="retry_with_backoff",
            attempt_number=attempt_number,
            success=True,
            provider=provider,
            backoff_seconds=backoff_seconds
        )

        return {
            "strategy": "retry_with_backoff",
            "success": True,
            "provider": provider,
            "attempt_number": attempt_number,
            "backoff_seconds": backoff_seconds,
            "message": f"Retrying {provider} in {backoff_seconds} seconds (attempt {attempt_number})"
        }

    def _handle_circuit_breaker(
        self,
        provider: str,
        exception: Exception,
        correlation_id: str
    ) -> Dict[str, Any]:
        # Handle circuit breaker activation

        # Mark provider as temporarily unavailable
        if provider in self.active_providers:
            self.active_providers.remove(provider)

        log_recovery_attempt(
            correlation_id=correlation_id,
            strategy="circuit_breaker",
            attempt_number=1,
            success=True,
            provider=provider,
            failure_count=self.provider_failure_counts[provider]
        )

        return {
            "strategy": "circuit_breaker",
            "success": True,
            "provider": provider,
            "message": f"Circuit breaker activated for {provider}",
            "recovery_time_seconds": 60,  # Standard circuit breaker recovery time
            "degradation_level": self._update_degradation_level()
        }

    def _handle_fail_fast(
        self,
        provider: str,
        exception: Exception,
        correlation_id: str
    ) -> Dict[str, Any]:
        # Handle terminal failure - no recovery possible

        log_recovery_attempt(
            correlation_id=correlation_id,
            strategy="fail_fast",
            attempt_number=1,
            success=False,
            provider=provider,
            error=str(exception)
        )

        return {
            "strategy": "fail_fast",
            "success": False,
            "provider": provider,
            "message": f"Terminal failure for {provider}: {exception}",
            "degradation_level": self.current_degradation_level.value
        }

    def _get_next_available_provider(self, failed_provider: str) -> Optional[str]:
        # Get next available provider for fallback

        # Check configured fallback providers first
        for fallback in self.config.fallback_providers:
            if (fallback != failed_provider and
                fallback not in self.failed_providers and
                self.provider_failure_counts.get(fallback, 0) == 0):
                return fallback

        # Fall back to any available provider
        for provider in self.active_providers:
            if (provider != failed_provider and
                self.provider_failure_counts.get(provider, 0) < self.config.max_retry_attempts):
                return provider

        return None

    def _calculate_degradation_level(self, total_failures: int) -> DegradationLevel:
        # Calculate appropriate degradation level based on failure count

        if total_failures >= self.config.emergency_mode_threshold:
            return DegradationLevel.EMERGENCY_MODE
        elif total_failures >= self.config.minimal_service_threshold:
            return DegradationLevel.MINIMAL_SERVICE
        elif total_failures >= self.config.reduced_functionality_threshold:
            return DegradationLevel.REDUCED_FUNCTIONALITY
        else:
            return DegradationLevel.FULL_SERVICE

    def _update_degradation_level(self) -> str:
        # Update degradation level based on current provider status
        active_count = len(self.active_providers)
        failed_count = len(self.failed_providers)

        if active_count == 0:
            self.current_degradation_level = DegradationLevel.EMERGENCY_MODE
        elif failed_count >= 2:
            self.current_degradation_level = DegradationLevel.MINIMAL_SERVICE
        elif failed_count >= 1:
            self.current_degradation_level = DegradationLevel.REDUCED_FUNCTIONALITY
        else:
            self.current_degradation_level = DegradationLevel.FULL_SERVICE

        return self.current_degradation_level.value

    def _get_degradation_message(self, level: DegradationLevel) -> str:
        # Get user-friendly message for degradation level
        messages = {
            DegradationLevel.FULL_SERVICE: "All systems operational",
            DegradationLevel.REDUCED_FUNCTIONALITY: "Some LLM providers unavailable - using fallback providers",
            DegradationLevel.MINIMAL_SERVICE: "Multiple provider failures - operating with reduced functionality",
            DegradationLevel.EMERGENCY_MODE: "Critical: All LLM providers failed - emergency mode active"
        }
        return messages.get(level, "Unknown degradation level")

    def _get_available_features(self, level: DegradationLevel) -> List[str]:
        # Get list of available features for degradation level
        feature_sets = {
            DegradationLevel.FULL_SERVICE: [
                "full_llm_functionality", "multi_provider_support", "advanced_retry_logic"
            ],
            DegradationLevel.REDUCED_FUNCTIONALITY: [
                "basic_llm_functionality", "single_provider", "standard_retry_logic"
            ],
            DegradationLevel.MINIMAL_SERVICE: [
                "basic_llm_functionality", "extended_timeouts", "simple_retry_logic"
            ],
            DegradationLevel.EMERGENCY_MODE: [
                "cached_responses_only", "manual_intervention_required"
            ]
        }
        return feature_sets.get(level, [])

    def _get_timeout_multiplier(self, level: DegradationLevel) -> float:
        # Get timeout multiplier for degradation level
        multipliers = {
            DegradationLevel.FULL_SERVICE: 1.0,
            DegradationLevel.REDUCED_FUNCTIONALITY: 1.5,
            DegradationLevel.MINIMAL_SERVICE: 2.0,
            DegradationLevel.EMERGENCY_MODE: 3.0
        }
        return multipliers.get(level, 1.0)

    def get_current_status(self) -> Dict[str, Any]:
        # Get current degradation status
        return {
            "degradation_level": self.current_degradation_level.value,
            "active_providers": self.active_providers,
            "failed_providers": self.failed_providers,
            "provider_failure_counts": self.provider_failure_counts,
            "available_features": self._get_available_features(self.current_degradation_level),
            "timeout_multiplier": self._get_timeout_multiplier(self.current_degradation_level)
        }

    def reset_provider_status(self, provider: str):
        # Reset failure status for a recovered provider
        if provider in self.failed_providers:
            self.failed_providers.remove(provider)
            if provider not in self.active_providers:
                self.active_providers.append(provider)

        self.provider_failure_counts[provider] = 0
        self._update_degradation_level()

        self.logger.info(f"Provider {provider} status reset - back to active")


# Global degradation manager instance
_degradation_manager = GracefulDegradationManager()


def handle_llm_provider_failure(
    provider: str,
    exception: Exception,
    context: Optional[ErrorContext] = None
) -> Dict[str, Any]:
    # Convenience function to handle LLM provider failures
    return _degradation_manager.handle_provider_failure(provider, exception, context)


def register_llm_providers(providers: List[str]):
    # Register available LLM providers for degradation management
    _degradation_manager.register_providers(providers)


def get_degradation_status() -> Dict[str, Any]:
    # Get current degradation status
    return _degradation_manager.get_current_status()


def reset_provider_status(provider: str):
    # Reset provider status after recovery
    _degradation_manager.reset_provider_status(provider)
