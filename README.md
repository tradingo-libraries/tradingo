# Tradingo
declarative signal and portfolio construction  framework  evaluated
on a dag.

Create a yaml file 'config.yaml' like so

  ```
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

  prices:
    include: "file://{{ TP_TEMPLATES }}/instruments.yaml"
    variables:
      include: "file://{{ TP_CONFIG_HOME }}/universes/im-multi-asset-3.yaml"
      start_date: "{{ data_interval_start }}"
      end_date: "{{ data_interval_end }}"

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

and `$TP_CONFIG_HOME/universes/im-multi-asset-3.yaml`
  ```
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

Set the following:
  1. TP_CONFIG_HOME
  2. TP_ARCTIC_URI
  3. IG_SERVICE_API_KEY
  4. IG_SERVICE_ACC_TYPE
  5. IG_SERVICE_USERNAME
  6. IG_SERVICE_PASSWORD

And run
  ```
  tradingo-cli --config ./config.yaml task run backtest --end-date "2025-04-04 00:00:00+00:00"
  ```
