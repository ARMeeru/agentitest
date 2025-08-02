"""
Secure Credential Management for AgentiTest Framework.

This module provides secure storage and retrieval of credentials including:
- Environment variable encryption
- Key derivation and secure storage
- Integration with external secret managers
- Credential rotation support
"""

import os
import json
import base64
import hashlib
from typing import Dict, Optional, Any, Union
from pathlib import Path
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import logging

logger = logging.getLogger(__name__)


class SecureCredentialManager:
    """
    Manages secure storage and retrieval of sensitive credentials.
    
    Features:
    - Encrypts credentials at rest
    - Supports multiple storage backends
    - Integrates with external secret managers
    - Provides credential rotation capabilities
    """
    
    def __init__(self, storage_path: str = ".credentials", master_key: Optional[str] = None):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(exist_ok=True)
        
        # Initialize encryption
        self._master_key = master_key or self._get_or_create_master_key()
        self._fernet = self._create_cipher()
        
        # Storage backends
        self._backends = {
            'file': self._file_backend,
            'env': self._env_backend,
        }
        
        # Try to import optional secret manager integrations
        self._init_secret_managers()
    
    def _get_or_create_master_key(self) -> str:
        """Get or create master encryption key."""
        key_file = self.storage_path / ".master_key"
        
        if key_file.exists():
            try:
                with open(key_file, 'r') as f:
                    return f.read().strip()
            except Exception as e:
                logger.warning(f"Could not read master key: {e}")
        
        # Generate new master key
        master_key = base64.urlsafe_b64encode(os.urandom(32)).decode()
        
        try:
            # Save master key with restricted permissions
            key_file.touch(mode=0o600)
            with open(key_file, 'w') as f:
                f.write(master_key)
            logger.info("Created new master encryption key")
        except Exception as e:
            logger.error(f"Could not save master key: {e}")
            # Use in-memory key as fallback
        
        return master_key
    
    def _create_cipher(self) -> Fernet:
        """Create cipher for encryption/decryption."""
        # Derive key from master key
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=b'agentitest_salt',  # In production, use random salt
            iterations=100000,
        )
        key = base64.urlsafe_b64encode(kdf.derive(self._master_key.encode()))
        return Fernet(key)
    
    def _init_secret_managers(self):
        """Initialize optional secret manager integrations."""
        # AWS Secrets Manager
        try:
            import boto3
            self._backends['aws'] = self._aws_secrets_backend
            logger.debug("AWS Secrets Manager integration available")
        except ImportError:
            pass
        
        # Azure Key Vault
        try:
            from azure.keyvault.secrets import SecretClient
            from azure.identity import DefaultAzureCredential
            self._backends['azure'] = self._azure_keyvault_backend
            logger.debug("Azure Key Vault integration available")
        except ImportError:
            pass
        
        # HashiCorp Vault
        try:
            import hvac
            self._backends['vault'] = self._hashicorp_vault_backend
            logger.debug("HashiCorp Vault integration available")
        except ImportError:
            pass
    
    def store_credential(
        self, 
        key: str, 
        value: str, 
        backend: str = 'file',
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Store a credential securely.
        
        Args:
            key: Credential identifier
            value: Credential value
            backend: Storage backend ('file', 'env', 'aws', 'azure', 'vault')
            metadata: Additional metadata
        
        Returns:
            Success status
        """
        try:
            if backend not in self._backends:
                raise ValueError(f"Unknown backend: {backend}")
            
            # Encrypt the credential
            encrypted_value = self._fernet.encrypt(value.encode()).decode()
            
            # Create credential record
            credential_record = {
                'key': key,
                'encrypted_value': encrypted_value,
                'backend': backend,
                'metadata': metadata or {},
                'created_at': self._get_timestamp(),
                'updated_at': self._get_timestamp()
            }
            
            # Store using backend
            success = self._backends[backend]('store', key, credential_record)
            
            if success:
                logger.info(f"Stored credential '{key}' using {backend} backend")
            else:
                logger.error(f"Failed to store credential '{key}' using {backend} backend")
            
            return success
            
        except Exception as e:
            logger.error(f"Error storing credential '{key}': {e}")
            return False
    
    def retrieve_credential(
        self, 
        key: str, 
        backend: str = 'file',
        fallback_backends: Optional[list] = None
    ) -> Optional[str]:
        """
        Retrieve a credential.
        
        Args:
            key: Credential identifier
            backend: Primary storage backend
            fallback_backends: List of fallback backends to try
        
        Returns:
            Decrypted credential value or None
        """
        backends_to_try = [backend]
        if fallback_backends:
            backends_to_try.extend(fallback_backends)
        
        for backend_name in backends_to_try:
            try:
                if backend_name not in self._backends:
                    continue
                
                # Retrieve using backend
                credential_record = self._backends[backend_name]('retrieve', key, None)
                
                if credential_record:
                    # Decrypt the credential
                    encrypted_value = credential_record['encrypted_value']
                    decrypted_value = self._fernet.decrypt(encrypted_value.encode()).decode()
                    
                    logger.debug(f"Retrieved credential '{key}' from {backend_name} backend")
                    return decrypted_value
                    
            except Exception as e:
                logger.warning(f"Error retrieving credential '{key}' from {backend_name}: {e}")
                continue
        
        logger.warning(f"Could not retrieve credential '{key}' from any backend")
        return None
    
    def list_credentials(self, backend: str = 'file') -> list:
        """List all stored credentials for a backend."""
        try:
            if backend not in self._backends:
                return []
            
            return self._backends[backend]('list', None, None) or []
            
        except Exception as e:
            logger.error(f"Error listing credentials from {backend}: {e}")
            return []
    
    def delete_credential(self, key: str, backend: str = 'file') -> bool:
        """Delete a credential."""
        try:
            if backend not in self._backends:
                return False
            
            success = self._backends[backend]('delete', key, None)
            
            if success:
                logger.info(f"Deleted credential '{key}' from {backend} backend")
            
            return success
            
        except Exception as e:
            logger.error(f"Error deleting credential '{key}': {e}")
            return False
    
    def rotate_credential(
        self, 
        key: str, 
        new_value: str, 
        backend: str = 'file'
    ) -> bool:
        """
        Rotate a credential by updating its value.
        
        Args:
            key: Credential identifier
            new_value: New credential value
            backend: Storage backend
        
        Returns:
            Success status
        """
        # Store old value as backup
        old_value = self.retrieve_credential(key, backend)
        if old_value:
            backup_key = f"{key}_backup_{self._get_timestamp()}"
            self.store_credential(backup_key, old_value, backend)
        
        # Store new value
        return self.store_credential(key, new_value, backend)
    
    # Backend implementations
    
    def _file_backend(self, operation: str, key: str, data: Any) -> Any:
        """File-based storage backend."""
        credentials_file = self.storage_path / "credentials.json"
        
        # Load existing credentials
        credentials = {}
        if credentials_file.exists():
            try:
                with open(credentials_file, 'r') as f:
                    credentials = json.load(f)
            except Exception as e:
                logger.warning(f"Could not load credentials file: {e}")
        
        if operation == 'store':
            credentials[key] = data
            try:
                credentials_file.touch(mode=0o600)
                with open(credentials_file, 'w') as f:
                    json.dump(credentials, f, indent=2)
                return True
            except Exception as e:
                logger.error(f"Could not save credentials file: {e}")
                return False
        
        elif operation == 'retrieve':
            return credentials.get(key)
        
        elif operation == 'list':
            return list(credentials.keys())
        
        elif operation == 'delete':
            if key in credentials:
                del credentials[key]
                try:
                    with open(credentials_file, 'w') as f:
                        json.dump(credentials, f, indent=2)
                    return True
                except Exception:
                    return False
            return True
        
        return None
    
    def _env_backend(self, operation: str, key: str, data: Any) -> Any:
        """Environment variable backend."""
        env_key = f"AGENTITEST_CRED_{key.upper()}"
        
        if operation == 'store':
            os.environ[env_key] = json.dumps(data)
            return True
        
        elif operation == 'retrieve':
            env_value = os.getenv(env_key)
            if env_value:
                try:
                    return json.loads(env_value)
                except json.JSONDecodeError:
                    return None
            return None
        
        elif operation == 'list':
            prefix = "AGENTITEST_CRED_"
            return [k[len(prefix):].lower() for k in os.environ.keys() if k.startswith(prefix)]
        
        elif operation == 'delete':
            if env_key in os.environ:
                del os.environ[env_key]
            return True
        
        return None
    
    def _aws_secrets_backend(self, operation: str, key: str, data: Any) -> Any:
        """AWS Secrets Manager backend."""
        import boto3
        from botocore.exceptions import ClientError
        
        client = boto3.client('secretsmanager')
        secret_name = f"agentitest/{key}"
        
        if operation == 'store':
            try:
                # Try to update existing secret
                client.update_secret(
                    SecretId=secret_name,
                    SecretString=json.dumps(data)
                )
                return True
            except ClientError as e:
                if e.response['Error']['Code'] == 'ResourceNotFoundException':
                    # Create new secret
                    try:
                        client.create_secret(
                            Name=secret_name,
                            SecretString=json.dumps(data)
                        )
                        return True
                    except ClientError:
                        return False
                return False
        
        elif operation == 'retrieve':
            try:
                response = client.get_secret_value(SecretId=secret_name)
                return json.loads(response['SecretString'])
            except ClientError:
                return None
        
        elif operation == 'list':
            try:
                response = client.list_secrets(
                    Filters=[
                        {
                            'Key': 'name',
                            'Values': ['agentitest/']
                        }
                    ]
                )
                return [s['Name'].replace('agentitest/', '') for s in response['SecretList']]
            except ClientError:
                return []
        
        elif operation == 'delete':
            try:
                client.delete_secret(SecretId=secret_name, ForceDeleteWithoutRecovery=True)
                return True
            except ClientError:
                return False
        
        return None
    
    def _azure_keyvault_backend(self, operation: str, key: str, data: Any) -> Any:
        """Azure Key Vault backend."""
        from azure.keyvault.secrets import SecretClient
        from azure.identity import DefaultAzureCredential
        from azure.core.exceptions import ResourceNotFoundError
        
        vault_url = os.getenv("AZURE_KEY_VAULT_URL")
        if not vault_url:
            logger.error("AZURE_KEY_VAULT_URL not configured")
            return None
        
        credential = DefaultAzureCredential()
        client = SecretClient(vault_url=vault_url, credential=credential)
        
        secret_name = f"agentitest-{key}"
        
        if operation == 'store':
            try:
                client.set_secret(secret_name, json.dumps(data))
                return True
            except Exception:
                return False
        
        elif operation == 'retrieve':
            try:
                secret = client.get_secret(secret_name)
                return json.loads(secret.value)
            except ResourceNotFoundError:
                return None
        
        elif operation == 'list':
            try:
                secrets = client.list_properties_of_secrets()
                return [s.name.replace('agentitest-', '') for s in secrets if s.name.startswith('agentitest-')]
            except Exception:
                return []
        
        elif operation == 'delete':
            try:
                client.begin_delete_secret(secret_name)
                return True
            except Exception:
                return False
        
        return None
    
    def _hashicorp_vault_backend(self, operation: str, key: str, data: Any) -> Any:
        """HashiCorp Vault backend."""
        import hvac
        
        vault_url = os.getenv("VAULT_URL", "http://localhost:8200")
        vault_token = os.getenv("VAULT_TOKEN")
        
        if not vault_token:
            logger.error("VAULT_TOKEN not configured")
            return None
        
        client = hvac.Client(url=vault_url, token=vault_token)
        
        if not client.is_authenticated():
            logger.error("Vault authentication failed")
            return None
        
        secret_path = f"secret/agentitest/{key}"
        
        if operation == 'store':
            try:
                client.secrets.kv.v2.create_or_update_secret(
                    path=f"agentitest/{key}",
                    secret=data
                )
                return True
            except Exception:
                return False
        
        elif operation == 'retrieve':
            try:
                response = client.secrets.kv.v2.read_secret_version(path=f"agentitest/{key}")
                return response['data']['data']
            except Exception:
                return None
        
        elif operation == 'list':
            try:
                response = client.secrets.kv.v2.list_secrets(path="agentitest")
                return response['data']['keys']
            except Exception:
                return []
        
        elif operation == 'delete':
            try:
                client.secrets.kv.v2.delete_metadata_and_all_versions(path=f"agentitest/{key}")
                return True
            except Exception:
                return False
        
        return None
    
    def _get_timestamp(self) -> str:
        """Get current timestamp as string."""
        from datetime import datetime
        return datetime.utcnow().isoformat()


class CredentialLoader:
    """
    Loads credentials from various sources with fallback support.
    
    Supports loading from:
    1. Secure credential manager
    2. Environment variables
    3. Configuration files
    4. External secret managers
    """
    
    def __init__(self, credential_manager: Optional[SecureCredentialManager] = None):
        self.credential_manager = credential_manager or SecureCredentialManager()
        self.cache = {}
        self.cache_ttl = 300  # 5 minutes
    
    def get_credential(
        self, 
        key: str, 
        required: bool = True,
        fallback_env_var: Optional[str] = None,
        default: Optional[str] = None
    ) -> Optional[str]:
        """
        Get credential with fallback support.
        
        Priority order:
        1. Secure credential manager
        2. Environment variable (fallback_env_var or key)
        3. Default value
        
        Args:
            key: Credential key
            required: Whether credential is required
            fallback_env_var: Fallback environment variable name
            default: Default value if not found
        
        Returns:
            Credential value or None
        """
        # Check cache first
        cache_key = f"{key}:{fallback_env_var or 'none'}"
        cached_value, timestamp = self.cache.get(cache_key, (None, 0))
        
        if cached_value and (self._get_timestamp() - timestamp) < self.cache_ttl:
            return cached_value
        
        # Try secure credential manager
        value = self.credential_manager.retrieve_credential(
            key, 
            backend='file',
            fallback_backends=['env', 'aws', 'azure', 'vault']
        )
        
        if value:
            self.cache[cache_key] = (value, self._get_timestamp())
            return value
        
        # Try environment variable
        env_var = fallback_env_var or key
        value = os.getenv(env_var)
        
        if value and value != "YOUR_API_KEY":  # Ignore placeholder values
            self.cache[cache_key] = (value, self._get_timestamp())
            return value
        
        # Use default value
        if default:
            return default
        
        # Handle required credentials
        if required:
            raise ValueError(f"Required credential '{key}' not found")
        
        return None
    
    def _get_timestamp(self) -> float:
        """Get current timestamp."""
        import time
        return time.time()


# Global instances
_credential_manager = None
_credential_loader = None


def get_credential_manager() -> SecureCredentialManager:
    """Get global credential manager instance."""
    global _credential_manager
    if _credential_manager is None:
        _credential_manager = SecureCredentialManager()
    return _credential_manager


def get_credential_loader() -> CredentialLoader:
    """Get global credential loader instance."""
    global _credential_loader
    if _credential_loader is None:
        _credential_loader = CredentialLoader(get_credential_manager())
    return _credential_loader


def store_api_key(provider: str, api_key: str, backend: str = 'file') -> bool:
    """Convenience function to store API key."""
    manager = get_credential_manager()
    key = f"{provider}_api_key"
    return manager.store_credential(key, api_key, backend)


def get_api_key(provider: str) -> Optional[str]:
    """Convenience function to get API key."""
    loader = get_credential_loader()
    key = f"{provider}_api_key"
    env_var = f"{provider.upper()}_API_KEY"
    return loader.get_credential(key, required=False, fallback_env_var=env_var)