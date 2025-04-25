# Historical IG data may be downloaded from the following site.
# https://www.dukascopy.com/swiss/english/marketwatch/historical/
# https://forexsb.com/historical-forex-data
#
import argparse
import logging
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import List, Optional

import pandas as pd
from arcticdb import Arctic

import tradingo.sampling as sampling

logger = logging.getLogger(__name__)


if __name__ == "__main__":
    import sys

    logging.getLogger(__name__).setLevel(logging.INFO)

    env = os.environ.get("ENVIRONMENT", "dev")
    dir = Path(f"{Path.home()}/dev/tradingo-plat/data/{env}")
    dir.mkdir(parents=True, exist_ok=True)

    sys.argv.extend(
        [
            "--arctic-uri",
            f"lmdb://{dir}/tradingo.db",
            "--path",
            str(Path.home() / "dev" / "market-data" / "GAS"),
            str(Path.home() / "dev" / "market-data" / "USA500"),
            str(Path.home() / "dev" / "market-data" / "BRENT"),
            str(Path.home() / "dev" / "market-data" / "USTBOND"),
            str(Path.home() / "dev" / "market-data" / "COCOA"),
            # "--dry-run",
            "--clean",
            "--provider",
            "ig-trading",
            "--universe",
            "im-multi-asset",
            "--end-date",
            "2018-10-21 00:00:00+00:00",
        ]
    )

    main()
