# Tradingo

A functional backtesting framework for equities

There are three main configuration items

  * universe
  * signal_config
  * portfolio

A universe defines a list of symbols referenced from a file, a web link or a list of
tickers. They are associated with a provider.

A signal will depend on a universe, which is referenced by universe/provider key.

A portfolio references the same universe, and a set of signal configs by name
