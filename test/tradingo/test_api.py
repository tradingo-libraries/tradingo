import pytest

from tradingo.api import Tradingo


def test_tradingo_api(tradingo: Tradingo):

    df = tradingo.prices.close()


def test_tradingo_api_with(tradingo: Tradingo):

    t = tradingo

    t.instruments.etfs.concat.prices.close.transpose(axis=1).transpose()


def test_tradingo_merge(tradingo: Tradingo):
    tradingo.portfolio.model.raw.shares.concat.prices.close(
        axis=1, keys=("position", "close")
    )


if __name__ == "__main__":
    pytest.main()
