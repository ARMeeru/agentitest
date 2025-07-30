# AgentiTest Framework Architecture

## High-Level Flow Overview
```
┌─────────────┐    ┌──────────────┐    ┌─────────────┐    ┌──────────────┐
│   pytest    │───▶│  conftest.py │───▶│ browser-use │───▶│     LLM      │
│  execution  │    │   fixtures   │    │    Agent    │    │  providers   │
└─────────────┘    └──────────────┘    └─────────────┘    └──────────────┘
```

## Detailed Component Flow Diagram (Based on SauceDemo)

[![](https://mermaid.ink/img/pako:eNp9l1tTIjsQx79Kah6OL4or6IpTtadq5CYKiIC6OlpUHAKkHJLZTEZl1e9-Olcu6559sDbw76S70_1L8x4kfEKCMJgJnM3RqP7AEPyL4uucCCQKlocoW0qSS6T-jHNcJGRCFnz8q6DJcylbPqK9vX_RaWxVdZon_IWIJfoHDYkssscHZvY81cJanHA2VVKwBUWeU85Qk77JQpAcdTiePBp9TevrcSIIlmScposxZbnELCGoiRPJxdIq61rZeO90uuP-4PKmXW8MUIO9UMHZgjCJbrCg-Ckln86XhrL4mJEFZfQDNQ_i2hzLFuezlKC2PeRxXcozwrCSlrX0EpZR-2spZnIueEYTUFe0OnKf_MXgN4QO4kMjVqv_238m-C9QHxmfYbEuM8Lmgc5IK4aM-G9RTSfSpbdZNhq7qmysDjdWR25l1i29PIufBH-FKhlngk8pJM5eot3_TKva8alR9a3olco5mhM8SeHqf4xEQfabOM1Xvre13XlMVvc3FiTjQkJBbh5xrqUXcYswIiA0tGZTAq8yIiSFoppygaI0NYbG9EKbduKRqtkukXM-QY03khRSleNQYiFze0pHS7s-3HyzZrfS2tXqngvbFbgOu5_i5augs7n0fvS0_FL7MVS9VYfeOsU5TUqm3-DqJlhMxoU6OuUzyuxBl9qyH4OYRDOIWG1ResEpnah2kTh_RglOU-2ZMelrk6u4xqEkRJFINC3S1EjhA8pmdu8rLRzED0GLI8nRXMosD_f3X19fS54ApYQv9neRnBMGiVe3s7Ph7U6pVHoI_OEDvecwBqaMsfLXnDstWKJybk8eatUo1hFtla5RGd1I666tDgKa0lkh8NpO11pxE6tTwlWg441Ab7ToNga4hEjhg2523K3-_uf23Ydo83qt-qdW38UML0iIjGt0An_plBLhfb_TsvtYZ6EE-TDVwRn4RrIxYZMfgiQc0qjW3uxem0WRjXlVrR3OM-tBFBkan8ZDMEUHIerhFzpTvQHXeD3oWN2phXFNE2JAcM4ZZAXVoGD8gTVL4LoWRQyny5y6rqhb6DagSPRXvyFkUy0bFYKgIpCuW10LGmOaY6jZ1PvWSUJNTrdcRVMqculo1TQMasUDeFQEqGd8LPm4ECnCyUZptCyfzlwPokgLVhlzhLKIaserxtz0YiMSa9W2gDqP-3CB-sGC0lEBKs4s0AtEA0-Nd-fcUuoiXrtUdMb5MxrBkTMiPDsuLJU6saEV0pc40Garmu1YIHXjITyNhOVzDh2AszUwdi2FerZWRnNeQHDqeYUD3XE9C5_LWGU7khInc4VOBxhLmH5ct50FDuPk2XhiiWKRcmXqrRwCFdMUqeFBNYHDieXJ4Otq03SweBiul5NOcAoJJpNdi5jC7uyLaWiJMdoqpjbLCjWzvEl1kc4MaoqkLv6Rhci1rymqjMbaaKuori1ObraKKoQ54wv0ObxYvtz-efd7aO32_lEX48-6tdT5aZJasUnt4zx_hU0cbCxt7lTgKzcI7GrHNOfFnQXOvfM99HuZfMBf-05ozjjQRH947V2MPGgsaQ5DVEthIDSd4CnjMFMzTlrJZavdQ0-FlL4Va5419ZWTw-JpQaXuKg8dR52G6T7brPoi9il7gSKBsbA0l4vUw8bRpvn3cJoeLy0TzlGIboigU5hQiyQByD86tji4nJmIbK0imDYm8KDmKAO3PF8cYNpQ1X2nsICAChC67NCOxuM4NyfBS7Xjq7vtaXNuW7lJ4UToobxI5RpjHGQuHDpW7Rxu2xjUONZ04s2ZIcpzNTb5q-l43nTfzXdfOAydA9Go_f2I3TUQ-lCl9gGs6ZlZqx8Nh43646amiWn6AcAxcxBqRu1Oo25eRe9OQwi-ekN7Fl59wNP2YFZLCWaFexL7nlFXvndrKc9XcHJ0GrjcDfS4iexYuUrFwFNqaBxdvcE1vshSIsljsBssiFhgOoEfVe_K7iGAEUkRK4T_AiKeH4IH9gk6XEg-XLIkCGEYI7uBUJB2iyJTV1KnGH6YLdyHZEKhvrvmF5v-4bYbZJjdc-4lM6FOthvCKEFEjRdMBmH5sKzFQfgevAVhtVo6qnw7qZS_VQ8q1fLhbrAMwr3v1W-lk5Pj6tFJ-fj7Yfnw6HM3-K13P4LPq9VK5eT4uFI-Ojiufv4HF4eQnQ?type=png)](https://mermaid.live/edit#pako:eNp9l1tTIjsQx79Kah6OL4or6IpTtadq5CYKiIC6OlpUHAKkHJLZTEZl1e9-Olcu6559sDbw76S70_1L8x4kfEKCMJgJnM3RqP7AEPyL4uucCCQKlocoW0qSS6T-jHNcJGRCFnz8q6DJcylbPqK9vX_RaWxVdZon_IWIJfoHDYkssscHZvY81cJanHA2VVKwBUWeU85Qk77JQpAcdTiePBp9TevrcSIIlmScposxZbnELCGoiRPJxdIq61rZeO90uuP-4PKmXW8MUIO9UMHZgjCJbrCg-Ckln86XhrL4mJEFZfQDNQ_i2hzLFuezlKC2PeRxXcozwrCSlrX0EpZR-2spZnIueEYTUFe0OnKf_MXgN4QO4kMjVqv_238m-C9QHxmfYbEuM8Lmgc5IK4aM-G9RTSfSpbdZNhq7qmysDjdWR25l1i29PIufBH-FKhlngk8pJM5eot3_TKva8alR9a3olco5mhM8SeHqf4xEQfabOM1Xvre13XlMVvc3FiTjQkJBbh5xrqUXcYswIiA0tGZTAq8yIiSFoppygaI0NYbG9EKbduKRqtkukXM-QY03khRSleNQYiFze0pHS7s-3HyzZrfS2tXqngvbFbgOu5_i5augs7n0fvS0_FL7MVS9VYfeOsU5TUqm3-DqJlhMxoU6OuUzyuxBl9qyH4OYRDOIWG1ResEpnah2kTh_RglOU-2ZMelrk6u4xqEkRJFINC3S1EjhA8pmdu8rLRzED0GLI8nRXMosD_f3X19fS54ApYQv9neRnBMGiVe3s7Ph7U6pVHoI_OEDvecwBqaMsfLXnDstWKJybk8eatUo1hFtla5RGd1I666tDgKa0lkh8NpO11pxE6tTwlWg441Ab7ToNga4hEjhg2523K3-_uf23Ydo83qt-qdW38UML0iIjGt0An_plBLhfb_TsvtYZ6EE-TDVwRn4RrIxYZMfgiQc0qjW3uxem0WRjXlVrR3OM-tBFBkan8ZDMEUHIerhFzpTvQHXeD3oWN2phXFNE2JAcM4ZZAXVoGD8gTVL4LoWRQyny5y6rqhb6DagSPRXvyFkUy0bFYKgIpCuW10LGmOaY6jZ1PvWSUJNTrdcRVMqculo1TQMasUDeFQEqGd8LPm4ECnCyUZptCyfzlwPokgLVhlzhLKIaserxtz0YiMSa9W2gDqP-3CB-sGC0lEBKs4s0AtEA0-Nd-fcUuoiXrtUdMb5MxrBkTMiPDsuLJU6saEV0pc40Garmu1YIHXjITyNhOVzDh2AszUwdi2FerZWRnNeQHDqeYUD3XE9C5_LWGU7khInc4VOBxhLmH5ct50FDuPk2XhiiWKRcmXqrRwCFdMUqeFBNYHDieXJ4Otq03SweBiul5NOcAoJJpNdi5jC7uyLaWiJMdoqpjbLCjWzvEl1kc4MaoqkLv6Rhci1rymqjMbaaKuori1ObraKKoQ54wv0ObxYvtz-efd7aO32_lEX48-6tdT5aZJasUnt4zx_hU0cbCxt7lTgKzcI7GrHNOfFnQXOvfM99HuZfMBf-05ozjjQRH947V2MPGgsaQ5DVEthIDSd4CnjMFMzTlrJZavdQ0-FlL4Va5419ZWTw-JpQaXuKg8dR52G6T7brPoi9il7gSKBsbA0l4vUw8bRpvn3cJoeLy0TzlGIboigU5hQiyQByD86tji4nJmIbK0imDYm8KDmKAO3PF8cYNpQ1X2nsICAChC67NCOxuM4NyfBS7Xjq7vtaXNuW7lJ4UToobxI5RpjHGQuHDpW7Rxu2xjUONZ04s2ZIcpzNTb5q-l43nTfzXdfOAydA9Go_f2I3TUQ-lCl9gGs6ZlZqx8Nh43646amiWn6AcAxcxBqRu1Oo25eRe9OQwi-ekN7Fl59wNP2YFZLCWaFexL7nlFXvndrKc9XcHJ0GrjcDfS4iexYuUrFwFNqaBxdvcE1vshSIsljsBssiFhgOoEfVe_K7iGAEUkRK4T_AiKeH4IH9gk6XEg-XLIkCGEYI7uBUJB2iyJTV1KnGH6YLdyHZEKhvrvmF5v-4bYbZJjdc-4lM6FOthvCKEFEjRdMBmH5sKzFQfgevAVhtVo6qnw7qZS_VQ8q1fLhbrAMwr3v1W-lk5Pj6tFJ-fj7Yfnw6HM3-K13P4LPq9VK5eT4uFI-Ojiufv4HF4eQnQ)

## Step-by-Step Execution Flow

### Phase 1: pytest Initialization
```
1. pytest test_saucedemo_quick.py
   ├── Discover test files
   ├── Load pytest.ini configuration
   ├── Set alluredir=allure-results
   ├── Enable asyncio mode
   └── Initialize logging
```

### Phase 2: Session Fixtures Setup (conftest.py)
```
2. Session-scoped fixtures execution:
   ├── create_llm_instance()
   │   ├── Read LLM_PROVIDER env var
   │   ├── Validate API keys
   │   ├── Create provider-specific LLM
   │   └── Return configured LLM instance
   │
   ├── browser_profile()
   │   ├── Read HEADLESS env var
   │   └── Return BrowserProfile config
   │
   └── environment_reporter()
       ├── Collect system info
       ├── Get LLM provider details
       ├── Get browser version info
       └── Write environment.properties
```

### Phase 3: Test Method Execution
```
3. test_standard_user_login():
   ├── Function-scoped browser_session fixture
   │   ├── Create BrowserSession(browser_profile)
   │   ├── Initialize Playwright browser
   │   └── Yield session to test
   │
   ├── BaseAgentTest.validate_task()
   │   ├── Construct full task string
   │   ├── Call run_agent_task()
   │   └── Perform assertion validation
   │
   └── Cleanup: browser_session.close()
```

### Phase 4: Agent Execution Loop
```
4. Agent.run() with browser-use:

   Each Agent Step:
   ├── Page Analysis
   │   ├── Take screenshot
   │   ├── Extract interactive elements
   │   ├── Build page state
   │   └── Create visual context
   │
   ├── LLM Reasoning Call
   │   ├── Send: task + page state + screenshot
   │   ├── LLM processes multimodal input
   │   ├── LLM generates reasoning
   │   └── LLM returns action decision
   │
   ├── Action Execution
   │   ├── Parse LLM response
   │   ├── Execute browser action
   │   ├── Wait for page state change
   │   └── Capture result
   │
   └── Step Recording (record_step hook)
       ├── Extract agent thoughts
       ├── Capture screenshot
       ├── Log URL and duration
       ├── Create Allure step
       └── Attach all artifacts
```

### Phase 5: LLM Provider Interaction Detail
```
5. LLM Processing (per step):

   Input to LLM:
   ├── Task instruction text
   ├── Current page screenshot (base64)
   ├── Interactive elements list
   ├── Previous action context
   └── Agent memory/history

   LLM Processing:
   ├── Vision model analyzes screenshot
   ├── Language model processes instructions
   ├── Reasoning about next action
   ├── Decision making process
   └── Action formulation

   LLM Response:
   ├── Thought process explanation
   ├── Action type (click, input, navigate)
   ├── Target element identification
   ├── Action parameters
   └── Success evaluation criteria
```

### Phase 6: Browser Action Execution
```
6. Browser Control via Playwright:

   Action Types:
   ├── go_to_url(url)
   │   ├── Navigate to specified URL
   │   ├── Wait for page load
   │   └── Update page state
   │
   ├── input_text(element, text)
   │   ├── Find element by index/selector
   │   ├── Clear existing content
   │   ├── Type new text
   │   └── Trigger input events
   │
   ├── click_element(element)
   │   ├── Find clickable element
   │   ├── Ensure element is visible
   │   ├── Perform click action
   │   └── Wait for response
   │
   └── take_screenshot()
       ├── Capture full page
       ├── Convert to base64
       └── Return image data
```

### Phase 7: Allure Integration
```
7. Real-time Reporting:

   Per Step:
   ├── allure.step(action_name)
   │   ├── Create step container
   │   ├── Set step title/description
   │   └── Track execution time
   │
   ├── Attachments
   │   ├── allure.attach(screenshot, PNG)
   │   ├── allure.attach(agent_thoughts, TEXT)
   │   ├── allure.attach(url, URI_LIST)
   │   └── allure.attach(duration, TEXT)
   │
   └── Step Status
       ├── Mark step as passed/failed
       ├── Record execution metrics
       └── Close step container

   Final Result:
   ├── allure.attach(final_result, TEXT)
   ├── Test status determination
   └── Report data serialization
```

### Phase 8: Test Validation & Cleanup
```
8. Test Completion:

   Validation:
   ├── Agent returns final result text
   ├── validate_task checks for expected substring
   ├── Assert passes/fails based on match
   └── Test status determined

   Cleanup:
   ├── browser_session.close()
   │   ├── Close browser tabs
   │   ├── Terminate browser process
   │   └── Clean temporary files
   │
   ├── Allure finalization
   │   ├── Write test result JSON
   │   ├── Save all attachments
   │   └── Update environment.properties
   │
   └── pytest result reporting
       ├── Log test outcome
       ├── Display execution stats
       └── Generate final report
```

## Key Framework Components & Their Roles

### 1. **conftest.py - Configuration Hub**
- **Session fixtures**: LLM, browser profile, environment
- **Helper functions**: Agent task execution, step recording
- **Base test class**: Common validation patterns

### 2. **browser-use Agent - Task Orchestrator**
- **Task interpretation**: Natural language to actions
- **Browser control**: Playwright integration
- **State management**: Memory and context tracking

### 3. **LLM Provider - Decision Engine**
- **Visual analysis**: Screenshot interpretation
- **Action planning**: Step-by-step reasoning
- **Error handling**: Adaptive problem solving

### 4. **Allure Integration - Observability Layer**
- **Step tracking**: Granular execution monitoring
- **Artifact capture**: Screenshots, logs, metadata
- **Report generation**: Interactive test reports

### 5. **Playwright Browser - Execution Engine**
- **Page interaction**: Click, type, navigate
- **State capture**: Screenshots, DOM analysis
- **Session management**: Browser lifecycle

## Data Flow Summary

```
Test Instructions (Natural Language)
    ↓
Agent Task Processing (browser-use)
    ↓
LLM Reasoning (Multi-modal AI)
    ↓
Browser Actions (Playwright)
    ↓
Page State Changes (DOM/Visual)
    ↓
Result Validation (Assertion)
    ↓
Allure Reporting (Rich Artifacts)
```

This architecture demonstrates how the framework seamlessly integrates multiple technologies to provide intelligent, observable, and maintainable browser automation testing.
