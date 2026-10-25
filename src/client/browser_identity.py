"""Browser persistence for the local Streamlit demo identity.

The token is readable by page JavaScript, so this is intentionally limited to a
trusted local demo. Production deployments should use server sessions and an
HttpOnly, Secure cookie issued after real authentication.
"""

from streamlit.components.v2 import component

_IDENTITY_COMPONENT = component(
    "saas_browser_identity",
    js="""
    export default function(component) {
      const { data, setStateValue } = component;
      const key = "saas_support_token";
      if (data && data.token) {
        localStorage.setItem(key, data.token);
        document.cookie = `${key}=${encodeURIComponent(data.token)}; Path=/; SameSite=Strict; Max-Age=31536000`;
        setStateValue("token", data.token);
      } else {
        const token = localStorage.getItem(key) || "";
        if (token) {
          document.cookie = `${key}=${encodeURIComponent(token)}; Path=/; SameSite=Strict; Max-Age=31536000`;
        }
        setStateValue("token", token);
      }
    }
    """,
)


def persist_browser_token(token: str) -> None:
    _IDENTITY_COMPONENT(
        key="saas-browser-identity",
        data={"token": token},
        default={"token": token},
        on_token_change=lambda: None,
        height=0,
    )
