"""CoordAlign - A tool to assign mean values to survey coordinates that do not fall
perfectly on the coordinate grid of a measured spatial variable.

Author: T.D. Medina
"""

import argparse
from functools import partial
from itertools import product, batched
from multiprocessing import set_start_method, Pool
from pathlib import Path

import pandas as pd
from pandas import IndexSlice as idx

from shared_memory_arrays import SharedPandasDataFrame
from tqdm import tqdm

class CustomHelp(argparse.HelpFormatter):
    """Custom help formatter_class that only displays metavar once."""

    def _format_action_invocation(self, action):
        if not action.option_strings:
            default = self._get_default_metavar_for_positional(action)
            metavar, = self._metavar_formatter(action, default)(1)
            return metavar
        parts = []
        if action.nargs == 0:
            parts.extend(action.option_strings)
        else:
            default = self._get_default_metavar_for_optional(action)
            args_string = self._format_args(action, default)
            for option_string in action.option_strings:
                parts.append("%s" % (option_string))
            parts[-1] += " %s " % args_string
        return ", ".join(parts)

    def _format_action(self, action):
        parts = super()._format_action(action)
        if action.nargs == argparse.PARSER:
            parts = "\n".join(parts.split("\n")[1:])
        return parts


set_start_method("fork")
_AXES = ["lon", "lat"]


def _date_option_fixer(date_opt):
    if date_opt is None:
        return slice(None)
    if isinstance(date_opt, list):
        return date_opt
    if isinstance(date_opt, int):
        return [date_opt]
    if isinstance(date_opt, tuple):
        return list(date_opt)
    raise TypeError(f"Unexpected type ({type(date_opt)}) for date variable: {date_opt}")


def read_coord_data(file_path, year=None, month=None):
    """Read variable data."""
    coord_data = pd.read_csv(file_path, sep=",")
    if "Year" not in coord_data.columns:
        coord_data["Year"] = 0
    if "Month" not in coord_data.columns:
        coord_data["Month"] = 0
    coord_data = (coord_data.set_index(["Year", "Month", "lon", "lat"])
                  .sort_index())
    if year is None and month is None:
        return coord_data
    year, month = _date_option_fixer(year), _date_option_fixer(month)
    coord_data = coord_data.loc[idx[year, month, :, :],]
    return coord_data


def read_station_table(file_path):
    """Read survey positions."""
    station_table = pd.read_csv(file_path)
    station_table.set_index(["lon", "lat"], inplace=True)
    return station_table


def _average_closest_subtask(coordinates: list,
                             coord_table: SharedPandasDataFrame,
                             lonlat: dict,
                             variable_name: str):
    """Find the coordinate grid tile of each survey coordinate and assign mean value.

    This is a subtask intended for multiprocessing with shared memory locations.
    """
    coord_table = coord_table.read()
    lonlat = {axis: shared_df.read() for axis, shared_df in lonlat.items()}
    results, missing, incomplete = [], [], []
    setsize = {4}

    for coordinate in coordinates:
        points = [(lonlat[axis] - coordinate[axis])
                  .abs().nsmallest(2, axis).index.to_list()
                  for axis in _AXES]

        try:
            result = [coord_table.loc[idx[:, :, point[0], point[1]],]
                      for point in product(*points)]
        except KeyError:
            missing.append(coordinate)
            continue

        result = pd.concat(result).groupby(["Year", "Month"])
        if set(result.size()) != setsize:
            incomplete.append(coordinate)
            continue

        result = pd.concat([result.mean(), result.max() - result.min()],
                           axis=1)
        result.columns = [f"{variable_name}_mean", f"{variable_name}_range"]
        result["lon"], result["lat"] = coordinate["lon"], coordinate["lat"]
        result = result.reset_index().set_index(_AXES)
        results.append(result)
    results = pd.concat(results)
    missing = pd.DataFrame(missing)
    if not missing.empty:
        missing.set_index(_AXES, inplace=True)
    incomplete = pd.DataFrame(incomplete)
    if not incomplete.empty:
        incomplete.set_index(_AXES, inplace=True)
    return results, missing, incomplete


def _average_closest_multiprocess(station_table, coordinate_data, variable_name,
                                  subtask_length=None):
    """Find the coordinate grid tile of each survey coordinate and assign mean value.

    This function utilizes multiprocessing and shared memory locations to parallel
    process the data without terrible overhead.
    """
    lonlat = {axis: coordinate_data.index.unique(axis).to_frame() for axis in _AXES}
    coordinates = [coord._asdict() for coord
                   in station_table.reset_index()[_AXES].itertuples(index=False)]
    if subtask_length is None:
        subtask_length = len(coordinates) // 8 + 1
    coordinates = batched(coordinates, subtask_length)
    try:
        coordinate_data = SharedPandasDataFrame(coordinate_data)
        lonlat = {axis: SharedPandasDataFrame(frame) for axis, frame in lonlat.items()}

        with Pool() as pool:
            func = partial(_average_closest_subtask,
                           coord_table=coordinate_data,
                           lonlat=lonlat,
                           variable_name=variable_name)
            results = list(pool.imap_unordered(func, coordinates))
    finally:
        coordinate_data.unlink()
        for thing in lonlat.values():
            thing.unlink()
    results = [pd.concat([result[i] for result in results]) for i in range(3)]
    for i, result in enumerate(results):
        if result.empty and result.index.empty:
            continue
        results[i] = result.merge(station_table, left_index=True, right_index=True)
    return results


def average_closest(station_table, coordinate_data, variable_name):
    """Find the coordinate grid tile of each survey coordinate and assign mean value."""
    results, missing = [], []
    setsize = {4}
    total = station_table.shape[0]
    lonlat = {axis: coordinate_data.index.unique(axis).to_series() for axis in _AXES}
    for i, coordinate in tqdm(enumerate(station_table.reset_index().itertuples()),
                              desc=f"Locating stations", total=total,
                              unit="Station"):
        points = [(lonlat[axis] - coordinate._asdict()[axis])
                  .abs().nsmallest(2).index.to_list()
                  for axis in _AXES]

        try:
            result = [coordinate_data.loc[idx[:, :, point[0], point[1]],]
                      for point in product(*points)]
        except KeyError:
            missing.append(coordinate)
            continue

        result = pd.concat(result).groupby(["Year", "Month"])
        result = pd.concat([result.mean(), result.max() - result.min(), result.size()],
                           axis=1)
        result.columns = [f"{variable_name}_mean", f"{variable_name}_range",
                          f"{variable_name}_pointcount"]
        result["lon"], result["lat"] = coordinate.lon, coordinate.lat
        result = result.reset_index().set_index(_AXES)
        results.append(result)
    results = pd.concat(results).merge(station_table, left_index=True, right_index=True)
    missing = pd.DataFrame(missing)
    if not missing.empty:
        missing.set_index(_AXES, inplace=True)
    return results, missing


def main(station_data_file, measurement_data_file, variable_name,
         variable_year=None, variable_month=None,
         output_prefix=None, output_suffix=None,
         multiprocess=False):
    if output_prefix is None:
        output_prefix = Path(station_data_file).stem
        output_suffix = Path(measurement_data_file).stem
    stations = read_station_table(station_data_file)
    data = read_coord_data(measurement_data_file, variable_year, variable_month)
    if multiprocess:
        results, missing, incomplete = _average_closest_multiprocess(stations, data,
                                                                     variable_name)
    else:
        results, missing = average_closest(stations, data, variable_name)
    results.to_csv(".".join([output_prefix, output_suffix, "csv"]), index=True)
    if not missing.empty:
        missing.to_csv(".".join([output_prefix, output_suffix, "missing", "csv"]),
                     index=True)
    return results, missing


def _setup_argparse():
    parser = argparse.ArgumentParser()
    parser.formatter_class = CustomHelp
    station_args = parser.add_argument_group(title="Station Data")
    station_args.add_argument("-s", "--station-file", required=True,
                              dest="station_data_file",
                              help="File path to station CSV file. Must have columns "
                                   "called 'lon' and 'lat'. Additional columns are "
                                   "allowed but must not match columns in the desired "
                                   "variable data CSV.")

    variable_args = parser.add_argument_group(title="Variable Data")
    variable_args.add_argument("-v", "--variable-data-file", required=True,
                               dest="measurement_data_file",
                               help="File path to variable CSV file. Must have columns "
                                    "called 'lon' and 'lat' as well as a column with "
                                    "the desired variable measurement. Additional "
                                    "columns are ignored.")
    variable_args.add_argument("-vn", "--variable-name", required=True,
                               dest="variable_name",
                               help="Name of the desired variable column.")
    variable_args.add_argument("-y", "--year", type=int,
                               dest="variable_year", nargs="*",
                               help="Subset variable data to only these years. Multiple "
                                    "years are allowed, space-separated.")
    variable_args.add_argument("-m", "--month", type=int,
                               dest="variable_month", nargs="*",
                               help="Subset variable data to only these months. "
                                    "Multiple months are allowed, space-separated.")

    output_args = parser.add_argument_group(
        title="Output Options",
        description="Options to change the output file names. Default output file is "
                    "'./<station-file>.<variable-file>[.missing].csv'")
    output_args.add_argument("-op", "--output-prefix",
                             help="Prefix to use when writing output results. "
                                  "Default='./<station-file>'")
    output_args.add_argument("-os", "--output-suffix",
                             help="Suffix to use when writing output results. "
                                  "Default='<variable-file>[.<year>][.month]'")

    parser.add_argument("-x", "--multiprocess", action="store_true",
                        help="Enable parallel multiprocessing. May increase speed. May "
                             "not work on Windows operating systems.")
    return parser


if __name__ == '__main__':
    import sys
    parser = _setup_argparse()
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(0)
    args = parser.parse_args()

    results = main(**vars(args))
