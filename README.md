# Tradingo

Declarative signal and portfolio construction framework evaluated on a dag.

## Configuration

Tradingo is a declarative framework: configs are its core. A configuration represent a set of tasks executed in one DAG. Typically, a task is an ETL made of dependencies, function, parameters and output. While dependencies or output might be omitted, a function and its parameters (if any) are always needed. A task configuration looks like this:

```yaml
tasks-stage:
  tasks-stage.my-config:
    # Task Dependencies and Symbol Management
    depends_on: [ ... ]
    symbols_out: [ ... ]
    symbols_in:  [ ... ]
    # Function Definition
    function: "path.to.my.function"
    params: { ... }
```

An end-to-end configuration requires configs for all stages:

1. 🌌 a universe of symbols [and timing or sampling intervals]
2. 🛠️ price sampling via a data provider
3. 📈 one/many signals
4. 🧪 a portfolio construction recipe
5. 🎯 a trading (execution) strategy

_NOTE_: points 2, 4, 5 above use templates provided in `tradingo.templates` declared through the `include` pattern and populated with the `variables` (at runtime).

* 2 `tradingo.templates.instruments/*.yaml` specific to data providers (e.g. IG, Yahoo Finance, etc.)
* 4 `tradingo.templates.portfolio_construction.yaml` declares how a portfolio is build (depends on some signals)
* 5 `tradingo.templates.downstream_tasks.yaml` declares how a portfolio is traded (depends on a portfolio and a universe)

The signals is not templated, as it is specific to your strategy and likely proprietary.

### 1. Universe

Create a universe `$TP_CONFIG_HOME/universes/im-multi-asset-3.yaml` like so:

```yaml
universe_name: im-multi-asset-3
raw_prices_lib: prices_igtrading
epics:
  - "CC.D.NG.UMP.IP"
  - "IX.D.SPTRD.IFS.IP"
  - "CC.D.LCO.UMP.IP"
  - "IR.D.10YEAR100.FWM2.IP"
  - "CC.D.CC.UMP.IP"
  - "CC.D.S.UMP.IP"
  - "IX.D.NASDAQ.IFS.IP"
  - "CS.D.CFPGOLD.CFP.IP"
interval: 15min
start_date: "2017-01-01 00:00:00+00:00"
end_date: "{{ data_interval_end }}"
```

### 2. Price Sampling

Inspect package resources at `tradingo.templates.instruments` to see the sampling structure:

* 2.1 Download universe of symbols: `download.{{ universe_name }}`
 fetches the static instruments for the specified universe, saving them as a pickle file in instruments/{{ universe_name }}. This step is optional as it can be done once.

* 2.2 Sample each symbol: `sample.{{ symbol }}`
by using `tradingo.sampling.<provider>.<function>` API to collect price data for the defined date range and interval, storing it in `{{ raw_prices_lib }}/{{ symbol }}` (e.g. `prices/AAPL.ohlcv`, where `.ohlcv` is a suffix we introduced to be more expressive of the content)

* 2.3 Combine all sampled prices into one universe dataset: `sample.{{ universe_name }}`
 by using `tradingo.sampling.<provider>.<function>`, data from individual symbol samples can be rearranged depending on its format (e.g. {ask, mid, bid} for each symbol’s {open, high, low, close} can be published under `prices/my-universe.mid.close`, etc.)

NOTE: symbols are sometimes called tickers, epics, symbols, etc depending on the convention of the data provider.

### 3. Signals

This is the most free-style part of the config.

* 3.1 Task Dependencies and Symbol Management. this is equal to all other tasks.

* 3.2 Function Definition. this is custom to the function you define (assuming it can be imported) and the parameters it accepts.

```yaml
signals:
  signals.my-strategy:
    # Task Dependencies and Symbol Management
    depends_on: ["sample.my-universe"]
    symbols_out:
      - "signals/my-strategy.sig1"
      - "signals/my-strategy.sig2"
      - "signals/my-strategy.sig3"
    symbols_in:
      bid_close: "prices/my-universe.bid.close"
      ask_close: "prices/my-universe.ask.close"
    # Function Definition
    function: "path.to.my.function"
    params:
      fast_speed: 10
      slow_speed: 50
```

### 4. Portfolio Construction

Inspect package resources at `tradingo.templates.portfolio_construction.yaml` to see the portfolio construction template:

* 4.1 Build the portfolio: `portfolio.{{ name }}`
it assembles the porfolio to be traded from the signals, scale it and rounds it accounting for the AUM.

* 4.2 Build the portfolio backtest: `backtest.{{ name }}`
given the portfolio, produces main metrics (PnL, exposure, etc.). This is optional as it isn't used for trading, but necessary for portfolio management.

* 4.3 : `portfolio.point_in_time.{{ name }}`
given the portfolio, produces a end-of-run snapshot. This is optional as it isn't used for trading, but necessary for portfolio management.

### 5. Trading

Inspect package resources at `tradingo.templates.downstream_tasks.yaml` to see the trading template:

This piece of code takes a portfolio's position and it trades them against a provider.

### Putting Config Together

For the universe, create `$TP_CONFIG_HOME/universes/im-multi-asset-3.yaml` like so:

```yaml
universe_name: im-multi-asset-3
raw_prices_lib: prices_igtrading
epics:
  - "CC.D.NG.UMP.IP"
  - "IX.D.SPTRD.IFS.IP"
  - "CC.D.LCO.UMP.IP"
  - "IR.D.10YEAR100.FWM2.IP"
  - "CC.D.CC.UMP.IP"
  - "CC.D.S.UMP.IP"
  - "IX.D.NASDAQ.IFS.IP"
  - "CS.D.CFPGOLD.CFP.IP"
interval: 15min
start_date: "2017-01-01 00:00:00+00:00"
end_date: "{{ data_interval_end }}"
```

For the other configs, create a yaml file 'config.yaml' to populate tradingo templates (see below) like so:

```yaml

# 2. sampling
prices:
  include: "file://{{ TP_TEMPLATES }}/instruments/ig-trading.yaml"
  variables:
    include: "file://{{ TP_CONFIG_HOME }}/universes/im-multi-asset-3.yaml"
    start_date: "{{ data_interval_start }}"
    end_date: "{{ data_interval_end }}"

# 3. signals
signals:
  signals.intraday_momentum.im-multi-asset-3:
    depends_on: ["sample.im-multi-asset-3"]
    function: "tradingo.signals.intraday_momentum"
    symbols_out:
      - "signals/intraday_momentum.im-multi-asset-3"
      - "signals/intraday_momentum.z_score.im-multi-asset-3"
      - "signals/intraday_momentum.short_vol.im-multi-asset-3"
      - "signals/intraday_momentum.long_vol.im-multi-asset-3"
      - "signals/intraday_momentum.previous_close_px.im-multi-asset-3"
    symbols_in:
      bid_close: "prices/im-multi-asset-3.bid.close"
      ask_close: "prices/im-multi-asset-3.ask.close"
    params:
      calendar: "NYSE"
      frequency: "15min"
      long_vol: 64
      short_vol: 6
      threshold: 1
      cap: 2
      close_offset_periods: 1
      monotonic: true
      vol_floor_window: 120
      only_with_close: false
      close_overrides:
        CC.D.CC.UMP.IP:
          hours: 18
          minutes: 0
          timezone: "Europe/London"
        CC.D.S.UMP.IP:
          hours: 19
          minutes: 0
          timezone: "Europe/London"

portfolio:
  include: "file://{{ TP_TEMPLATES }}/portfolio_construction.yaml"
  variables:
    backtest:
      price_ffill_limit: 5
    universe_name: im-multi-asset-3
    name: intraday
    model_weights:
      "intraday_momentum.im-multi-asset-3": 1.0
    portfolio:
      aum: 10000
      multiplier: 5

trading:
  include: "file://{{ TP_TEMPLATES }}/downstream_tasks.yaml"
  variables:
    portfolio_name: intraday
    universe_name: im-multi-asset-3

```

Set the following env vars:
  0. `TP_CONFIG_HOME`
  1. `TP_TEMPLATES`
  2. `TP_ARCTIC_URI`
  3. `IG_SERVICE_API_KEY`
  4. `IG_SERVICE_ACC_TYPE`
  5. `IG_SERVICE_USERNAME`
  6. `IG_SERVICE_PASSWORD`

And run

```bash
tradingo-cli --auth --config "<path/to/config.yaml>" task run "<task-name>" --end-date "<date>"
```
