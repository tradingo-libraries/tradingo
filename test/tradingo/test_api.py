import pytest

from tradingo.api import Tradingo


def test_tradingo_api(tradingo: Tradingo) -> None:
    _ = tradingo.prices.close()


if __name__ == "__main__":
    pytest.main()
