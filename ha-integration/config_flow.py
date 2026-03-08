"""Config flow for LocalFresh Conversation."""

import voluptuous as vol
from homeassistant import config_entries

DOMAIN = "localfresh"


class LocalFreshConfigFlow(config_entries.ConfigFlow, domain=DOMAIN):
    """Config flow for LocalFresh."""

    VERSION = 1

    async def async_step_user(self, user_input=None):
        """Handle the initial step."""
        if user_input is not None:
            return self.async_create_entry(
                title=user_input.get("name", "LocalFresh"),
                data=user_input,
            )

        return self.async_show_form(
            step_id="user",
            data_schema=vol.Schema({
                vol.Required("name", default="Cedric"): str,
                vol.Required("url", default="http://192.168.0.69:8400"): str,
                vol.Required("token"): str,
                vol.Optional("model", default=""): str,
                vol.Optional("system_prompt", default=""): str,
            }),
        )
