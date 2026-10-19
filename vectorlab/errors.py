"""Errors for problems the user can fix."""


class UserError(Exception):
    """An expected, user-fixable problem such as a bad path or an invalid spec.

    The CLI prints the message without a traceback.
    """
