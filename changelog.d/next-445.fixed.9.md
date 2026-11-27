`dumps` refuses box data holding a dict whose keys are not strings, with its
usual 'not JSON serialisable' `TypeError`, instead of silently writing int,
float, bool and None keys as strings that `loads` cannot read back
([next#445](https://github.com/rel-int/discopy-next/issues/445)).
