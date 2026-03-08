"""Conversation agent for LocalFresh LLM."""

import asyncio
import json
import logging

import aiohttp
from homeassistant.components.conversation import (
    ConversationEntity,
    ConversationInput,
    ConversationResult,
)
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.helpers.entity_platform import AddEntitiesCallback
from homeassistant.helpers.intent import IntentResponse, IntentResponseErrorCode

from . import DOMAIN

_LOGGER = logging.getLogger(__name__)


async def async_setup_entry(
    hass: HomeAssistant, entry: ConfigEntry, async_add_entities: AddEntitiesCallback
) -> None:
    """Set up the LocalFresh conversation entity."""
    async_add_entities([LocalFreshConversationEntity(hass, entry)])


class LocalFreshConversationEntity(ConversationEntity):
    """LocalFresh conversation agent entity."""

    _attr_has_entity_name = True

    def __init__(self, hass: HomeAssistant, entry: ConfigEntry) -> None:
        """Initialize."""
        self.hass = hass
        self.entry = entry
        self._attr_name = entry.data.get("name", entry.title)
        self._attr_unique_id = entry.entry_id
        self._session_map: dict[str, str] = {}

    @property
    def supported_languages(self) -> list[str]:
        """Return supported languages."""
        return ["en"]

    async def async_process(
        self, user_input: ConversationInput
    ) -> ConversationResult:
        """Process a sentence."""
        config = self.hass.data[DOMAIN][self.entry.entry_id]
        url = config["url"].rstrip("/")
        token = config["token"]
        model = config.get("model", "")
        system_prompt = self.entry.data.get("system_prompt", "")

        conv_id = user_input.conversation_id or ""
        session_id = self._session_map.get(conv_id, "")

        payload = {
            "message": user_input.text,
            "enable_tools": True,
        }
        if session_id:
            payload["session_id"] = session_id
        if model:
            payload["model"] = model
        if system_prompt:
            payload["system_prompt"] = system_prompt

        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        }

        response_text = ""
        new_session_id = ""

        try:
            timeout = aiohttp.ClientTimeout(
                total=300, sock_connect=10, sock_read=120
            )
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(
                    f"{url}/v1/chat",
                    json=payload,
                    headers=headers,
                ) as resp:
                    if resp.status != 200:
                        body = await resp.text()
                        _LOGGER.error(
                            "LocalFresh API error %s: %s", resp.status, body[:200]
                        )
                        return _error_result(
                            user_input, f"API error: {resp.status}"
                        )

                    # Read full response body then parse SSE events.
                    # The stream has "token" events (incremental) and a
                    # final "done" event with the full text. Use the
                    # done event's text to avoid duplication.
                    body = await resp.text()
                    current_event = ""
                    for line in body.split("\n"):
                        line = line.strip()
                        if line.startswith("event: "):
                            current_event = line[7:]
                        elif line.startswith("data: "):
                            try:
                                data = json.loads(line[6:])
                            except json.JSONDecodeError:
                                continue
                            if current_event == "done" and "text" in data:
                                response_text = data["text"]
                            elif current_event == "token" and "text" in data:
                                response_text += data["text"]
                            if "session_id" in data:
                                new_session_id = data["session_id"]

        except aiohttp.ClientError as err:
            _LOGGER.error("LocalFresh connection error: %s", err)
            return _error_result(user_input, "Connection error")
        except (TimeoutError, asyncio.CancelledError) as err:
            _LOGGER.error("LocalFresh request error: %s", err)
            if response_text:
                pass  # Use what we have
            else:
                return _error_result(user_input, "Request timed out")

        if not response_text:
            response_text = "I didn't get a response. Please try again."

        if new_session_id and conv_id:
            self._session_map[conv_id] = new_session_id

        intent_response = IntentResponse(language="en")
        intent_response.async_set_speech(response_text.strip())
        return ConversationResult(
            response=intent_response,
            conversation_id=conv_id or new_session_id,
        )


def _error_result(
    user_input: ConversationInput, error_msg: str
) -> ConversationResult:
    """Create an error conversation result."""
    intent_response = IntentResponse(language="en")
    intent_response.async_set_error(
        IntentResponseErrorCode.UNKNOWN, error_msg
    )
    return ConversationResult(
        response=intent_response,
        conversation_id=user_input.conversation_id or "",
    )
