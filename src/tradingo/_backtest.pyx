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
    float opening_position,
    float opening_avg_price,
    float[:] trades,
    float[:] prices,
    float[:] dividends,
):

    cdef size_t num_days = trades.shape[0]
    cdef size_t idx, idx_prev
 
    cdef float[:] unrealised_pnl = cvarray(shape=(num_days,), itemsize=sizeof(float), format="f")
    cdef float[:] realised_pnl = cvarray(shape=(num_days,), itemsize=sizeof(float), format="f")
    cdef float[:] total_pnl = cvarray(shape=(num_days,), itemsize=sizeof(float), format="f")
    cdef float[:] net_investment = cvarray(shape=(num_days,), itemsize=sizeof(float), format="f")
    cdef float[:] net_position = cvarray(shape=(num_days,), itemsize=sizeof(float), format="f")
    cdef float[:] avg_open_price = cvarray(shape=(num_days,), itemsize=sizeof(float), format="f")

    unrealised_pnl[0] = 0
    realised_pnl[0] = 0
    total_pnl[0] = 0
    net_investment[0] = 0
    net_position[0] = 0
    avg_open_price[0] = 0

    net_position[0] = opening_position
    avg_open_price[0] = opening_avg_price

    # transient output variables
    cdef float m_net_position = opening_position
    cdef float m_avg_open_price = opening_avg_price
    cdef float m_unrealised_pnl = 0
    cdef float m_total_pnl = 0
    cdef float m_realised_pnl = 0
    cdef float m_net_investment = 0

    # loop variables
    cdef float price
    cdef float trade_quantity

    for idx in range(1, num_days):

        idx_prev = idx - 1

        trade_quantity = trades[idx]
        dividend = dividends[idx]
        price = prices[idx]
        m_unrealised_pnl = (price - avg_open_price[idx_prev]) * net_position[idx_prev]

        m_net_position = net_position[idx_prev]
        m_realised_pnl = realised_pnl[idx_prev]
        m_avg_open_price = avg_open_price[idx_prev]

        if trade_quantity != 0 and not isnan(trade_quantity):

            # net investment
            m_net_investment = max(
                m_net_investment, abs(m_net_position * m_avg_open_price)
            )
            # realized pnl
            if abs(m_net_position + trade_quantity) < abs(m_net_position):
                m_realised_pnl += (
                    (price - m_avg_open_price) * abs(trade_quantity) * sign(m_net_position)
                )

            # avg open price
            if abs(m_net_position + trade_quantity) > abs(m_net_position):
                m_avg_open_price = (
                    (m_avg_open_price * m_net_position) + (price * trade_quantity)
                ) / (m_net_position + trade_quantity)
            else:
                # Check if it is close-and-open
                if trade_quantity > abs(m_net_position):
                    m_avg_open_price = price

        if dividend != 0 and not isnan(dividend):
            m_realised_pnl += m_net_position * dividend

        # total pnl
        m_total_pnl = m_realised_pnl + m_unrealised_pnl
        m_net_position += trade_quantity

        unrealised_pnl[idx] = m_unrealised_pnl
        realised_pnl[idx] = m_realised_pnl
        total_pnl[idx] = m_total_pnl
        net_investment[idx] = m_net_investment
        net_position[idx] = m_net_position
        avg_open_price[idx] = m_avg_open_price

    return np.column_stack(
        (
            np.asarray(unrealised_pnl),
            np.asarray(realised_pnl),
            np.asarray(total_pnl),
            np.asarray(net_investment),
            np.asarray(net_position),
            np.asarray(avg_open_price),
        )
    )
