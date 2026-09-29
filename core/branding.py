"""Centralised microcopy and voice.

Ease-of-use is the #1 property of this app, so everything speaks in one warm,
gentle, lightly-apologetic voice. To re-skin the whole app, change the constants
here — nothing else references the name or emoji directly.
"""

APP_NAME = "Lumi"
APP_EMOJI = "💛"
APP_TAGLINE = "your gentle sales analyst"
APP_SUBTITLE = "Intelligent Sales Analytics Platform"

# The four analytics layers, described warmly for the landing page.
LAYERS = [
    ("🌸 Descriptive", "What happened?",
     "Clear revenue KPIs, trends and breakdowns — at a glance."),
    ("🔍 Diagnostic", "Why did it happen?",
     "Customer segments and gentle anomaly flags."),
    ("🔮 Predictive", "What's coming next?",
     "A revenue forecast that's explained, never a black box."),
    ("💡 Prescriptive", "What should we do?",
     "Kind, ranked suggestions — each one shows its 'why'."),
]


def oops(what: str) -> str:
    """Apologetic error lead-in."""
    return f"oh no — {what} sorry about that! 🙈"


def cheer(what: str) -> str:
    """Warm success lead-in."""
    return f"yay, {what} ✨"


def gentle_empty(what: str) -> str:
    """Soft empty-state message."""
    return f"nothing here yet — {what}"
