from string import Formatter

from discopy import messages


def test_messages_format_positionally():
    """ Every message is filled with ``.format(*args)``, one per field. """
    for name, message in vars(messages).items():
        if not name.isupper():
            continue
        fields = [field for _, field, _, _ in Formatter().parse(message)
                  if field is not None]
        assert all(field == "" for field in fields), name
        assert message.format(*range(len(fields)))
