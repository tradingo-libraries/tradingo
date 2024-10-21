cimport cython
from libc.math cimport isnan, signbit
from cython.view cimport array as cvarray

cimport numpy as np
import numpy as np


cdef sign(float x):
    if signbit(x):
        return -1
    return 1


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef compute_backtest(
    float[:] trades,
    float[:] bid,
    float[:] ask,
    float[:] stop_limit,
    float[:] stop_loss,
    float[:] dividends,
):

    cdef size_t num_days = trades.shape[0]
    cdef size_t idx, idx_prev
 
    cdef float[:] unrealised_pnl = cvarray(shape=(num_days,), itemsize=sizeof(float), format="f")
    cdef float[:] realised_pnl = cvarray(shape=(num_days,), itemsize=sizeof(float), format="f")
    cdef float[:] total_pnl = cvarray(shape=(num_days,), itemsize=sizeof(float), format="f")
    cdef float[:] net_investment = cvarray(shape=(num_days,), itemsize=sizeof(float), format="f")
    cdef float[:] net_position = cvarray(shape=(num_days,), itemsize=sizeof(float), format="f")
    cdef float[:] net_exposure = cvarray(shape=(num_days,), itemsize=sizeof(float), format="f")
    cdef float[:] avg_open_price = cvarray(shape=(num_days,), itemsize=sizeof(float), format="f")
    cdef float[:] stop_trade = cvarray(shape=(num_days,), itemsize=sizeof(float), format="f")

    unrealised_pnl[0] = 0
    realised_pnl[0] = 0
    total_pnl[0] = 0
    net_investment[0] = 0
    net_position[0] = trades[0]
    if trades[0] != 0.0:
        avg_open_price[0] = bid[0] if trades[0] < 0 else ask[0]
    else:
        avg_open_price[0] = 0.0
    net_exposure[0] = trades[0] * avg_open_price[0]
    stop_trade[0] = 0

    # transient output variables
    cdef float m_net_position = 0
    cdef float m_avg_open_price = 0
    cdef float m_unrealised_pnl = 0
    cdef float m_total_pnl = 0
    cdef float m_realised_pnl = 0
    cdef float m_net_investment = 0
    cdef float m_loss_trade = 0
    cdef float m_net_exposure = 0

    # loop variables
    cdef float price
    cdef float trade_quantity
    cdef float trade_price
    cdef float c_stop_limit
    cdef float c_stop_loss

    for idx in range(1, num_days):

        idx_prev = idx - 1

        m_net_position = net_position[idx_prev]
        trade_quantity = trades[idx]
        c_stop_limit = stop_limit[idx]
        c_stop_loss = stop_loss[idx]
        m_loss_trade = 0
        price = (bid[idx] + ask[idx])/2

        if (
            (not isnan(c_stop_loss)) and (
                (m_net_position < 0.0 and price > c_stop_loss)
                or (m_net_position > 0.0 and price < c_stop_loss)
            )
        ) or (
            (not isnan(c_stop_limit)) and (
                (m_net_position < 0.0 and price < c_stop_limit)
                or (m_net_position > 0.0 and price > c_stop_limit)
            )
        ):
            m_loss_trade = -1 * m_net_position
            trade_quantity += m_loss_trade

        dividend = dividends[idx]
        m_unrealised_pnl = (price - avg_open_price[idx_prev]) * m_net_position 

        m_realised_pnl = realised_pnl[idx_prev]
        m_avg_open_price = avg_open_price[idx_prev] if m_net_position != 0 else 0.0

        if trade_quantity != 0 and not isnan(trade_quantity):

            trade_price = ask[idx] if trade_price > 0 else bid[idx]

            # net investment
            m_net_investment = max(
                m_net_investment, abs(m_net_position * m_avg_open_price)
            )
            # realized pnl
            if abs(m_net_position + trade_quantity) < abs(m_net_position):
                m_realised_pnl += (
                    (trade_price - m_avg_open_price) * abs(trade_quantity) * sign(m_net_position)
                )

            # avg open price
            if abs(m_net_position + trade_quantity) > abs(m_net_position):
                m_avg_open_price = (
                    (m_avg_open_price * m_net_position) + (trade_price * trade_quantity)
                ) / (m_net_position + trade_quantity)
            else:
                # Check if it is close-and-open
                if trade_quantity > abs(m_net_position):
                    m_avg_open_price = trade_price

        if dividend != 0 and not isnan(dividend):
            m_realised_pnl += m_net_position * dividend

        # total pnl
        m_total_pnl = m_realised_pnl + m_unrealised_pnl
        m_net_position += trade_quantity
        m_net_exposure = m_net_position * price

        unrealised_pnl[idx] = m_unrealised_pnl
        realised_pnl[idx] = m_realised_pnl
        total_pnl[idx] = m_total_pnl
        net_investment[idx] = m_net_investment
        net_position[idx] = m_net_position
        avg_open_price[idx] = m_avg_open_price
        stop_trade[idx] = m_loss_trade
        net_exposure[idx] = m_net_exposure

    return np.column_stack(
        (
            np.asarray(unrealised_pnl),
            np.asarray(realised_pnl),
            np.asarray(total_pnl),
            np.asarray(net_investment),
            np.asarray(net_position),
            np.asarray(net_exposure),
            np.asarray(avg_open_price),
            np.asarray(stop_trade),
        )
    )
