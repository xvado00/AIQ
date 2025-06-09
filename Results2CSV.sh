#!/bin/sh
# BASH script to convert ComuteFromLog output to a CSV format
sed s/\ +\\/-\ /,/ | sed s/\ SD\ /,/ | sed s/\ :/,\ / | sed s/\ \ log\\//,\ / | \
  sed s/_/\ / | sed s/_/,\ / | sed s/_/,\ / | \
  tr '(' ',' | tr ')' ',' | sed s/,_.*// | cut -c9-
