cimport cython
cimport numpy as np
from cython.view cimport array as cvarray
from libc.math cimport isnan, signbit

import numpy as np


cdef inline double sign(double x):
    if signbit(x):
        return -1
    return 1


cdef inline double mark(double price, double avg_open_price, double position):
    # mark-to-market of the position held after the bar's trade
    if position == 0.0:
        return 0.0
    return (price - avg_open_price) * position


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef compute_backtest(
    double[:] trades,
    double[:] bid,
    double[:] ask,
    double[:] stop_limit,
    double[:] stop_loss,
    double[:] dividends,
):

    cdef size_t num_days = trades.shape[0]
    cdef size_t idx, idx_prev
 
    cdef double[:] unrealised_pnl = cvarray(shape=(num_days,), itemsize=sizeof(double), format="d")
    cdef double[:] realised_pnl = cvarray(shape=(num_days,), itemsize=sizeof(double), format="d")
    cdef double[:] total_pnl = cvarray(shape=(num_days,), itemsize=sizeof(double), format="d")
    cdef double[:] net_investment = cvarray(shape=(num_days,), itemsize=sizeof(double), format="d")
    cdef double[:] net_position = cvarray(shape=(num_days,), itemsize=sizeof(double), format="d")
    cdef double[:] net_exposure = cvarray(shape=(num_days,), itemsize=sizeof(double), format="d")
    cdef double[:] avg_open_price = cvarray(shape=(num_days,), itemsize=sizeof(double), format="d")
    cdef double[:] stop_trade = cvarray(shape=(num_days,), itemsize=sizeof(double), format="d")

    realised_pnl[0] = 0
    net_investment[0] = 0
    net_position[0] = trades[0]
    if trades[0] != 0.0:
        avg_open_price[0] = bid[0] if trades[0] < 0 else ask[0]
    else:
        avg_open_price[0] = 0.0
    unrealised_pnl[0] = mark((bid[0] + ask[0]) / 2, avg_open_price[0], trades[0])
    total_pnl[0] = unrealised_pnl[0]
    net_exposure[0] = trades[0] * (bid[0] + ask[0]) / 2
    stop_trade[0] = 0

    # transient output variables
    cdef double m_net_position = 0
    cdef double m_avg_open_price = 0
    cdef double m_unrealised_pnl = 0
    cdef double m_total_pnl = 0
    cdef double m_realised_pnl = 0
    cdef double m_net_investment = 0
    cdef double m_loss_trade = 0
    cdef double m_net_exposure = 0

    # loop variables
    cdef double price
    cdef double trade_quantity
    cdef double trade_price
    cdef double c_stop_limit
    cdef double c_stop_loss
    cdef double closed
    cdef double dividend

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

        m_realised_pnl = realised_pnl[idx_prev]
        m_avg_open_price = avg_open_price[idx_prev] if m_net_position != 0 else 0.0

        if trade_quantity != 0 and not isnan(trade_quantity):

            trade_price = ask[idx] if trade_quantity > 0 else bid[idx]

            # net investment
            m_net_investment = max(
                m_net_investment, abs(m_net_position * m_avg_open_price)
            )
            if m_net_position != 0 and sign(trade_quantity) != sign(m_net_position):
                # realized pnl on the part of the position that is closed
                closed = min(abs(trade_quantity), abs(m_net_position))
                m_realised_pnl += (
                    (trade_price - m_avg_open_price) * closed * sign(m_net_position)
                )
                # close-and-open: the remainder opens at the trade price
                if abs(trade_quantity) > abs(m_net_position):
                    m_avg_open_price = trade_price
            else:
                # avg open price
                m_avg_open_price = (
                    (m_avg_open_price * m_net_position) + (trade_price * trade_quantity)
                ) / (m_net_position + trade_quantity)


        if dividend != 0 and not isnan(dividend):
            m_realised_pnl += m_net_position * dividend

        # total pnl
        m_net_position += trade_quantity
        m_unrealised_pnl = mark(price, m_avg_open_price, m_net_position)

        m_total_pnl = m_realised_pnl + m_unrealised_pnl

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
