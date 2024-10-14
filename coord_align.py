
import argparse
from itertools import product
from pathlib import Path
from warnings import warn

import numpy as np
import pandas as pd
from pandas import IndexSlice as idx


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


def read_coord_data(file_path, variable_name, year, month):
    coord_data = pd.read_csv(file_path, sep=",")
    coord_data.set_index(["Year", "Month"], inplace=True)
    coord_data = coord_data.loc[idx[year, month],].reset_index(drop=True)
    coord_data = coord_data.sort_values("lat").sort_values("lon")
    axes = ["lon", "lat"]
    for axis in axes:
        axis_table = pd.DataFrame(coord_data[axis].unique(), columns=[axis])
        axis_table.index.name = axis + "_index"
        axis_table.reset_index(inplace=True)
        coord_data = coord_data.merge(axis_table, on=axis)
    coord_data = coord_data[["lon", "lat", variable_name, "lon_index", "lat_index"]]
    coord_data.set_index(["lon_index", "lat_index"], inplace=True)
    return coord_data


def read_station_table(file_path, index_column=0):
    station_table = pd.read_csv(file_path, index_col=index_column)
    return station_table


def find_closest(station_table, coordinate_data, variable_name):
    results = []
    warns = []
    for coordinate in station_table.itertuples():
        diff_table = np.abs(coordinate_data[["lon", "lat"]]
                            - (coordinate.lon, coordinate.lat))
        points = [diff_table[axis]
                  .drop_duplicates()
                  .nsmallest(2)
                  .index.get_level_values(axis+"_index")
                  .to_list()
                  for axis in ["lon", "lat"]]
        try:
            result = pd.DataFrame([coordinate_data.loc[idx[point],]
                                   for point in product(*points)])
        except KeyError as errkey:
            warns.append(coordinate)
            continue
        points = sorted(result[["lon", "lat"]].itertuples(index=False, name=None))
        results.append([coordinate.Index, coordinate.lon, coordinate.lat, *points,
                        result[variable_name].mean(),
                        result[variable_name].max() - result[variable_name].min()])
    results = pd.DataFrame(results,
                           columns=[station_table.index.name, "lon", "lat",
                                    "vertex1", "vertex2", "vertex3", "vertex4",
                                    f"{variable_name}_mean", f"{variable_name}_range"])
    results.set_index(station_table.index.name, inplace=True)
    if warns:
        warn(f"Missing measurement at {len(warns)} coordinates.")
        warns = pd.DataFrame(warns)
        warns.set_index("Index", inplace=True)
        warns.index.name = station_table.index.name
    else:
        warns = None
    return results, warns


def main(station_data_file, measurement_data_file, variable_name,
         measurement_data_year, measurement_data_month,
         station_index_col=0, output_prefix=None, output_suffix=None):
    if output_prefix is None:
        output_prefix = Path(station_data_file).stem
        output_suffix = Path(measurement_data_file).stem
    stations = read_station_table(station_data_file, station_index_col)
    data = read_coord_data(measurement_data_file, variable_name, measurement_data_year,
                           measurement_data_month)
    closest, warns = find_closest(stations, data, variable_name=variable_name)
    stations = stations.merge(closest.drop(["lon", "lat"], axis=1),
                              left_index=True, right_index=True)
    stations.to_csv(".".join([output_prefix, output_suffix, "csv"]), index=True)
    warns.to_csv(".".join([output_prefix, output_suffix, "missing", "csv"]),
                 index=True)
    return stations, warns


def _setup_argparse():
    parser = argparse.ArgumentParser()
    parser.formatter_class = CustomHelp
    station_args = parser.add_argument_group(title="Station Data")
    station_args.add_argument("-s", "--station-file", required=True,
                              dest="station_data_file",
                              help="File path to station CSV file. Must have columns "
                                   "called 'lon' and 'lat', as well as a column with a "
                                   "unique index per row. Additional columns are "
                                   "allowed but must not match columns in the desired "
                                   "variable data CSV.")
    station_args.add_argument("-i", "--index-col", type=int, dest="station_index_col",
                              help="Integer corresponding to the column number ("
                                   "zero-indexed) of the index column in the station "
                                   "CSV file. [default=0]")

    variable_args = parser.add_argument_group(title="Variable Data")
    variable_args.add_argument("-d", "--variable-data-file", required=True,
                               dest="measurement_data_file",
                               help="File path to variable CSV file. Must have columns "
                                    "called 'lon', 'lat', 'Year', and 'Month' as well "
                                    "as a column with the desired variable measurement. "
                                    "Additional columns are ignored.")
    variable_args.add_argument("-v", "--variable", required=True,
                               dest="variable_name",
                               help="Name of the desired variable column.")
    variable_args.add_argument("-y", "--year", required=True, type=int,
                               dest="measurement_data_year",
                               help="Year to subset data.")
    variable_args.add_argument("-m", "--month", required=True, type=int,
                               dest="measurement_data_month",
                               help="Month to subset data.")

    output_args = parser.add_argument_group(
        title="Output Options",
        description="Options to change the output file names. Default output file is "
                    "'./<station-file>.<variable-file>[.missing].csv'")
    output_args.add_argument("-op", "--output-prefix",
                             help="Prefix to use when writing output results. "
                                  "[default='./<station-file>'")
    output_args.add_argument("-os", "--output-suffix",
                             help="Suffix to use when writing output results. "
                                  "[default='<variable-file>'")
    return parser


if __name__ == '__main__':
    import sys
    parser = _setup_argparse()
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(0)
    args = parser.parse_args()

    results = main(**vars(args))
    # data = read_coord_data("/home/tyler/Downloads/SBT.csv", 2022, 6)
    # stations = read_station_table("/home/tyler/Downloads/station_index.csv", 2)

