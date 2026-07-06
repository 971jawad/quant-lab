# Walk-forward results - 75 models, all out-of-sample (16y of 15-minute data)

Every number below comes from trades in test windows the model
never saw during parameter selection or fitting. Selection used
training data only; ML labels were embargoed at fold boundaries.

## Ranked by trade-level t-statistic

| model           |   n_trades |   win_rate |   avg_R |   t_stat |   profit_factor |   total_return_pct |   ann_return_pct |   sharpe |   max_dd_pct |   worst_day_pct |   trailing_dd_breaches |   oos_days | instrument   | strategy   | style   |
|:----------------|-----------:|-----------:|--------:|---------:|----------------:|-------------------:|-----------------:|---------:|-------------:|----------------:|-----------------------:|-----------:|:-------------|:-----------|:--------|
| EURUSD_smc_C    |        119 |     0.4202 | -0.0088 |    -0.07 |           0.986 |              -2.76 |            -0.17 |    -0.12 |        -5.35 |           -0.47 |                      2 |       4208 | EURUSD       | smc        | C       |
| XAUUSD_smc_B    |         82 |     0.3049 | -0.0246 |    -0.13 |           0.966 |              -0.49 |            -0.03 |    -0.01 |        -4.6  |           -0.56 |                      0 |       4101 | XAUUSD       | smc        | B       |
| EURUSD_smc_A    |        152 |     0.4211 | -0.0187 |    -0.18 |           0.966 |              -2.77 |            -0.17 |    -0.03 |       -10.89 |           -0.88 |                      2 |       4208 | EURUSD       | smc        | A       |
| MNQ_ml_err_B    |       4573 |     0.2439 | -0.0057 |    -0.2  |           0.993 |             -75.05 |            -8.56 |    -0.38 |       -93.1  |           -3.7  |                      4 |       3911 | MNQ          | ml_err     | B       |
| XAUUSD_smc_C    |         68 |     0.3971 | -0.0614 |    -0.41 |           0.903 |              -1.48 |            -0.09 |    -0.09 |        -4.36 |           -0.82 |                      0 |       4101 | XAUUSD       | smc        | C       |
| ES_smc_C        |        191 |     0.4084 | -0.0406 |    -0.45 |           0.935 |              -3.04 |            -0.2  |    -0.15 |        -5.45 |           -0.57 |                      2 |       3880 | ES           | smc        | C       |
| MNQ_ta_C        |        680 |     0.3485 | -0.031  |    -0.56 |           0.956 |              -9.48 |            -0.64 |    -0.12 |       -27.52 |           -1.49 |                      6 |       3887 | MNQ          | ta         | C       |
| ES_smc_B        |        220 |     0.4    | -0.0531 |    -0.64 |           0.916 |               0.34 |             0.02 |     0.05 |        -9.07 |           -1.42 |                      2 |       3868 | ES           | smc        | B       |
| ES_ta_C         |        584 |     0.3425 | -0.0408 |    -0.68 |           0.943 |              -3.88 |            -0.26 |    -0.08 |        -9.46 |           -1.35 |                     13 |       3889 | ES           | ta         | C       |
| MNQ_smc_B       |        115 |     0.287  | -0.1022 |    -0.69 |           0.865 |              -8.79 |            -0.62 |    -0.15 |       -14.59 |           -2.18 |                     14 |       3747 | MNQ          | smc        | B       |
| EURUSD_smc_B    |        208 |     0.3365 | -0.0708 |    -0.72 |           0.898 |              -3.7  |            -0.23 |    -0.05 |       -12.49 |           -2.08 |                      2 |       4182 | EURUSD       | smc        | B       |
| MES_smc_C       |        147 |     0.3878 | -0.1019 |    -1    |           0.845 |              -3.5  |            -0.23 |    -0.17 |        -5.57 |           -0.62 |                      2 |       3880 | MES          | smc        | C       |
| MES_ta_C        |        584 |     0.3425 | -0.0672 |    -1.11 |           0.908 |              -7.86 |            -0.53 |    -0.23 |       -10.04 |           -0.84 |                      7 |       3889 | MES          | ta         | C       |
| MNQ_ml_err_A    |       6763 |     0.3361 | -0.0269 |    -1.39 |           0.962 |             -84.17 |           -11.2  |    -0.34 |       -98.39 |           -3.63 |                      1 |       3911 | MNQ          | ml_err     | A       |
| MNQ_ta_B        |        568 |     0.3239 | -0.0916 |    -1.47 |           0.878 |             -19.12 |            -1.37 |    -0.37 |       -24.62 |           -2.11 |                      3 |       3887 | MNQ          | ta         | B       |
| MNQ_ml_B        |       5323 |     0.2642 | -0.0377 |    -1.5  |           0.953 |             -73.1  |            -8.11 |    -0.44 |       -85.6  |           -3.73 |                      4 |       3911 | MNQ          | ml         | B       |
| MNQ_smc_C       |         65 |     0.2769 | -0.2574 |    -1.58 |           0.662 |              -4.55 |            -0.31 |    -0.41 |        -5.1  |           -0.52 |                      1 |       3746 | MNQ          | smc        | C       |
| ES_smc_A        |         84 |     0.2619 | -0.2411 |    -1.59 |           0.668 |             -14.49 |            -1.01 |    -0.38 |       -16.19 |           -1.07 |                      2 |       3880 | ES           | smc        | A       |
| ES_ta_A         |        493 |     0.2921 | -0.1161 |    -1.63 |           0.845 |             -37.1  |            -2.96 |    -0.4  |       -45.13 |           -2.32 |                      2 |       3889 | ES           | ta         | A       |
| MES_smc_A       |         84 |     0.2619 | -0.2631 |    -1.73 |           0.646 |             -15.67 |            -1.1  |    -0.41 |       -17.2  |           -1.16 |                      1 |       3880 | MES          | smc        | A       |
| MES_smc_B       |        203 |     0.3645 | -0.1493 |    -1.74 |           0.779 |             -13.23 |            -0.92 |    -0.35 |       -17.73 |           -1.16 |                      1 |       3880 | MES          | smc        | B       |
| XAUUSD_ta_B     |        544 |     0.2261 | -0.1427 |    -1.84 |           0.829 |             -18.1  |            -1.2  |    -0.44 |       -22.29 |           -0.58 |                      5 |       4182 | XAUUSD       | ta         | B       |
| XAUUSD_ta_A     |        550 |     0.2836 | -0.13   |    -1.96 |           0.825 |             -43.67 |            -3.4  |    -0.47 |       -51.32 |           -1.74 |                      5 |       4182 | XAUUSD       | ta         | A       |
| MES_ta_A        |        493 |     0.2921 | -0.1435 |    -2.02 |           0.813 |             -43.17 |            -3.6  |    -0.5  |       -48.93 |           -2.34 |                      3 |       3889 | MES          | ta         | A       |
| MNQ_ta_A        |        592 |     0.2787 | -0.1373 |    -2.02 |           0.829 |             -48.08 |            -4.16 |    -0.49 |       -56.93 |           -2.73 |                      1 |       3887 | MNQ          | ta         | A       |
| MNQ_ml_rec_B    |       5355 |     0.2201 | -0.0578 |    -2.12 |           0.932 |             -91.37 |           -14.6  |    -0.72 |       -93.81 |           -3.99 |                      7 |       3911 | MNQ          | ml_rec     | B       |
| MNQ_smc_A       |        167 |     0.2874 | -0.2551 |    -2.55 |           0.633 |             -27.94 |            -2.18 |    -0.69 |       -29.57 |           -1.64 |                      3 |       3746 | MNQ          | smc        | A       |
| XAUUSD_ml_rec_B |       5898 |     0.2222 | -0.0662 |    -2.64 |           0.92  |             -93.12 |           -14.84 |    -0.67 |       -95.21 |           -3.32 |                      8 |       4199 | XAUUSD       | ml_rec     | B       |
| ES_ml_err_B     |       5044 |     0.2542 | -0.0681 |    -2.64 |           0.917 |             -79.84 |            -9.82 |    -0.56 |       -89.13 |           -3.61 |                      7 |       3904 | ES           | ml_err     | B       |
| ES_ta_B         |        483 |     0.2588 | -0.1931 |    -2.7  |           0.764 |             -40.38 |            -3.3  |    -0.5  |       -40.58 |           -3.08 |                      5 |       3889 | ES           | ta         | B       |
| XAUUSD_smc_A    |        172 |     0.3198 | -0.2529 |    -2.72 |           0.621 |             -28.37 |            -2.03 |    -0.67 |       -29.05 |           -1.53 |                      2 |       4100 | XAUUSD       | smc        | A       |
| XAUUSD_ta_C     |        710 |     0.3662 | -0.1318 |    -2.83 |           0.806 |             -18.94 |            -1.26 |    -0.58 |       -22.17 |           -1.28 |                      2 |       4181 | XAUUSD       | ta         | C       |
| MES_ta_B        |        483 |     0.2588 | -0.2205 |    -3.08 |           0.737 |             -47.16 |            -4.05 |    -0.67 |       -47.32 |           -2.34 |                      2 |       3889 | MES          | ta         | B       |
| EURUSD_ta_B     |        633 |     0.267  | -0.2043 |    -3.32 |           0.751 |             -59.96 |            -5.31 |    -0.62 |       -62.86 |           -3.69 |                      3 |       4231 | EURUSD       | ta         | B       |
| MES_ml_err_B    |       4890 |     0.2356 | -0.0911 |    -3.34 |           0.894 |             -92.88 |           -15.68 |    -0.92 |       -94.82 |           -3.79 |                      4 |       3904 | MES          | ml_err     | B       |
| XAUUSD_ml_B     |       5801 |     0.2396 | -0.0815 |    -3.38 |           0.9   |             -80.14 |            -9.24 |    -0.9  |       -81.9  |           -2.61 |                      6 |       4199 | XAUUSD       | ml         | B       |
| EURUSD_ta_C     |        774 |     0.3178 | -0.1884 |    -3.84 |           0.753 |             -25.59 |            -1.75 |    -0.81 |       -27.12 |           -1.12 |                      3 |       4231 | EURUSD       | ta         | C       |
| MNQ_ml_err_C    |       7756 |     0.3431 | -0.0685 |    -4.15 |           0.906 |             -64.37 |            -6.43 |    -0.72 |       -77.7  |           -2.01 |                      1 |       3911 | MNQ          | ml_err     | C       |
| MNQ_ml_A        |       7458 |     0.3248 | -0.0798 |    -4.46 |           0.888 |             -99.29 |           -27.31 |    -1.14 |       -99.58 |           -3.97 |                      1 |       3911 | MNQ          | ml         | A       |
| ES_ml_B         |       5524 |     0.2577 | -0.1095 |    -4.67 |           0.865 |             -97.45 |           -21.08 |    -1.11 |       -97.82 |           -3.96 |                      6 |       3904 | ES           | ml         | B       |
| EURUSD_ta_A     |        640 |     0.2578 | -0.2732 |    -4.68 |           0.658 |             -74.12 |            -7.74 |    -1.13 |       -75.03 |           -2.78 |                      2 |       4231 | EURUSD       | ta         | A       |
| XAUUSD_ml_A     |       7151 |     0.3146 | -0.0891 |    -4.93 |           0.873 |             -99.47 |           -27.01 |    -1.23 |       -99.59 |           -3.46 |                      1 |       4199 | XAUUSD       | ml         | A       |
| XAUUSD_ml_err_B |       5688 |     0.2273 | -0.1225 |    -5.08 |           0.852 |             -98.24 |           -21.54 |    -1.14 |       -98.66 |           -3.98 |                      6 |       4199 | XAUUSD       | ml_err     | B       |
| MNQ_ml_C        |       9216 |     0.3438 | -0.0791 |    -5.31 |           0.891 |             -92.61 |           -15.45 |    -1.33 |       -94.45 |           -3.12 |                      1 |       3911 | MNQ          | ml         | C       |
| ES_ml_rec_B     |       5640 |     0.2161 | -0.1388 |    -5.57 |           0.838 |             -99.73 |           -31.81 |    -1.41 |       -99.74 |           -4.13 |                      1 |       3904 | ES           | ml_rec     | B       |
| MNQ_ml_rec_A    |       8489 |     0.3162 | -0.1014 |    -6.01 |           0.86  |             -99.91 |           -36.3  |    -1.54 |       -99.93 |           -8.02 |                      1 |       3911 | MNQ          | ml_rec     | A       |
| MES_ml_B        |       5318 |     0.2424 | -0.148  |    -6.03 |           0.826 |             -98.75 |           -24.63 |    -1.5  |       -98.9  |           -6.21 |                      4 |       3904 | MES          | ml         | B       |
| XAUUSD_ml_err_A |       7211 |     0.3065 | -0.1108 |    -6.2  |           0.844 |             -99.84 |           -32.11 |    -1.52 |       -99.88 |           -3.76 |                     18 |       4199 | XAUUSD       | ml_err     | A       |
| ES_ml_err_A     |       6586 |     0.3079 | -0.1197 |    -6.27 |           0.838 |             -99.83 |           -33.63 |    -1.59 |       -99.89 |           -3.93 |                      3 |       3904 | ES           | ml_err     | A       |
| XAUUSD_ml_rec_A |       9221 |     0.3089 | -0.1002 |    -6.29 |           0.859 |             -99.95 |           -36.28 |    -1.52 |       -99.96 |           -3.74 |                      3 |       4199 | XAUUSD       | ml_rec     | A       |
| EURUSD_ml_err_B |       5716 |     0.2326 | -0.156  |    -6.56 |           0.817 |             -99.36 |           -25.89 |    -1.58 |       -99.37 |           -4.17 |                      2 |       4247 | EURUSD       | ml_err     | B       |
| EURUSD_ml_rec_B |       6340 |     0.2193 | -0.1557 |    -6.71 |           0.819 |             -98.66 |           -22.59 |    -1.37 |       -98.75 |           -3.47 |                      6 |       4247 | EURUSD       | ml_rec     | B       |
| MES_ml_rec_B    |       5640 |     0.2154 | -0.169  |    -6.82 |           0.808 |             -99.74 |           -31.88 |    -1.54 |       -99.77 |           -4.2  |                      1 |       3904 | MES          | ml_rec     | B       |
| EURUSD_ml_B     |       5845 |     0.2287 | -0.162  |    -6.9  |           0.81  |             -99.7  |           -29.19 |    -1.45 |       -99.73 |           -3.86 |                      1 |       4247 | EURUSD       | ml         | B       |
| ES_ml_A         |       7311 |     0.3104 | -0.128  |    -7.16 |           0.825 |             -99.94 |           -38.33 |    -1.8  |       -99.95 |           -3.96 |                      1 |       3904 | ES           | ml         | A       |
| ES_ml_err_C     |       8430 |     0.3473 | -0.1115 |    -7.36 |           0.847 |             -83.96 |           -11.14 |    -1.45 |       -85.34 |           -1.99 |                      1 |       3904 | ES           | ml_err     | C       |
| MES_ml_err_A    |       6532 |     0.3099 | -0.1488 |    -7.82 |           0.802 |             -99.96 |           -39.21 |    -1.97 |       -99.97 |           -3.85 |                      2 |       3904 | MES          | ml_err     | A       |
| XAUUSD_ml_C     |       9528 |     0.3471 | -0.1091 |    -7.97 |           0.845 |             -96.59 |           -18.36 |    -1.97 |       -96.71 |           -2.65 |                      1 |       4199 | XAUUSD       | ml         | C       |
| EURUSD_ml_A     |       7664 |     0.3167 | -0.1397 |    -8.12 |           0.808 |             -99.98 |           -39.63 |    -1.96 |       -99.98 |           -5.78 |                      5 |       4247 | EURUSD       | ml         | A       |
| ES_ml_rec_A     |       8455 |     0.3037 | -0.1368 |    -8.17 |           0.816 |             -99.99 |           -44.8  |    -2.01 |       -99.99 |           -3.87 |                      1 |       3904 | ES           | ml_rec     | A       |
| MNQ_ml_rec_C    |      11204 |     0.3332 | -0.1102 |    -8.19 |           0.851 |             -99.44 |           -28.42 |    -2.14 |       -99.52 |           -5.72 |                      1 |       3911 | MNQ          | ml_rec     | C       |
| XAUUSD_ml_rec_C |      12316 |     0.3406 | -0.1028 |    -8.38 |           0.855 |             -99.07 |           -24.49 |    -1.82 |       -99.19 |           -3.24 |                      2 |       4199 | XAUUSD       | ml_rec     | C       |
| MES_ml_A        |       7311 |     0.3081 | -0.157  |    -8.76 |           0.791 |             -99.99 |           -44.33 |    -2.2  |       -99.99 |           -4.66 |                      1 |       3904 | MES          | ml         | A       |
| EURUSD_ml_err_A |       7259 |     0.3082 | -0.1552 |    -8.8  |           0.789 |             -99.99 |           -41.02 |    -2.15 |       -99.99 |           -6.25 |                      3 |       4247 | EURUSD       | ml_err     | A       |
| XAUUSD_ml_err_C |       9015 |     0.3338 | -0.1282 |    -9.04 |           0.82  |             -95.35 |           -16.82 |    -1.95 |       -95.43 |           -2.3  |                      6 |       4199 | XAUUSD       | ml_err     | C       |
| ES_ml_C         |       9570 |     0.3196 | -0.1368 |    -9.32 |           0.82  |             -98.03 |           -22.4  |    -2.24 |       -98.13 |           -3.05 |                      1 |       3904 | ES           | ml         | C       |
| EURUSD_ml_C     |       9904 |     0.3342 | -0.1333 |    -9.45 |           0.822 |             -99.27 |           -25.31 |    -2.24 |       -99.32 |           -3.44 |                      3 |       4247 | EURUSD       | ml         | C       |
| MES_ml_err_C    |       8430 |     0.3471 | -0.1445 |    -9.52 |           0.807 |             -92.56 |           -15.44 |    -2.05 |       -93.01 |           -3.02 |                      4 |       3904 | MES          | ml_err     | C       |
| MES_ml_rec_A    |       8455 |     0.3014 | -0.1663 |    -9.91 |           0.782 |            -100    |           -51.04 |    -2.43 |      -100    |           -4.13 |                      1 |       3904 | MES          | ml_rec     | A       |
| EURUSD_ml_err_C |       9174 |     0.3285 | -0.1464 |    -9.95 |           0.806 |             -98.71 |           -22.73 |    -2.22 |       -98.81 |           -3.49 |                      2 |       4247 | EURUSD       | ml_err     | C       |
| EURUSD_ml_rec_A |       9865 |     0.3036 | -0.1601 |   -10.51 |           0.784 |            -100    |           -52.21 |    -2.55 |      -100    |           -3.84 |                      1 |       4247 | EURUSD       | ml_rec     | A       |
| MES_ml_C        |       9570 |     0.3196 | -0.1696 |   -11.52 |           0.783 |             -98.83 |           -24.94 |    -2.61 |       -98.88 |           -2.63 |                      2 |       3904 | MES          | ml         | C       |
| ES_ml_rec_C     |      11153 |     0.3257 | -0.1591 |   -12.14 |           0.788 |             -99.73 |           -31.79 |    -2.67 |       -99.74 |           -3.44 |                      1 |       3904 | ES           | ml_rec     | C       |
| MES_ml_rec_C    |      11148 |     0.3111 | -0.19   |   -14.04 |           0.759 |             -99.95 |           -39.08 |    -3.22 |       -99.95 |           -3.68 |                      1 |       3904 | MES          | ml_rec     | C       |
| EURUSD_ml_rec_C |      12965 |     0.316  | -0.1805 |   -14.72 |           0.765 |             -99.94 |           -35.53 |    -3.2  |       -99.94 |           -3.3  |                      2 |       4247 | EURUSD       | ml_rec     | C       |

## How to read this
- `t_stat` >= 2.0 -> edge unlikely to be luck. **None reached it.**
- `t_stat` 0.5-2.0 -> promising but unproven; needs more data.
- Negative -> the strategy lost after realistic costs.

## Aggregates by strategy family (mean across instruments/styles)

| strategy   |   t_stat |   avg_R |   profit_factor |   total_return_pct |   max_dd_pct |
|:-----------|---------:|--------:|----------------:|-------------------:|-------------:|
| ml         |   -6.632 |  -0.117 |           0.847 |            -95.543 |      -96.702 |
| ml_err     |   -5.888 |  -0.107 |           0.86  |            -90.941 |      -94.899 |
| ml_rec     |   -8.148 |  -0.133 |           0.829 |            -98.707 |      -99.033 |
| smc        |   -1.08  |  -0.127 |           0.819 |             -8.696 |      -12.481 |
| ta         |   -2.249 |  -0.141 |           0.82  |            -33.107 |      -38.088 |

## Aggregates by style

| style   |   t_stat |   avg_R |   profit_factor |   total_return_pct |   max_dd_pct |
|:--------|---------:|--------:|----------------:|-------------------:|-------------:|
| A       |   -5.035 |  -0.146 |           0.801 |            -72.708 |      -75.093 |
| B       |   -3.22  |  -0.113 |           0.861 |            -63.509 |      -67.298 |
| C       |   -6.144 |  -0.116 |           0.842 |            -59.98  |      -62.33  |

## Verification & caveats (read this before believing anything)

- **Data**: cross-verified against Nasdaq Inc. (SPY/NDX), LBMA gold PM
  fix, ECB EUR/USD reference rate, and micro-vs-mini contract arbitrage
  (all 7 checks VERIFIED, see data/verification_report.json).
- **No-lookahead**: signals use bar-close info only; entries next bar
  open; stops assumed to fill before targets intrabar; ML labels
  embargoed ML_HORIZON+1 bars at every fold boundary; parameters and
  risk configs selected on training windows only.
- **Noise canary passed**: pure-random features produced no edge beyond
  gold's unconditional drift (+0.099R on always-long) - the pipeline
  does not leak future data.
- **Drift benchmark**: XAUUSD ml_rec (the best family, t up to 1.97)
  earns +0.08..0.13R on longs - statistically indistinguishable from
  the drift baseline. Treat it as trend capture, not alpha.
- **The learning-from-mistakes scheme (ml_err)** cut index losses
  roughly in half vs uniform ML (t -1.4 -> -0.3 on MES) but did NOT
  turn them positive. Recency weighting (ml_rec) helped only on gold
  and hurt badly on indices (regime-chasing).
- **NO MODEL reached t-stat 2.0.** Nothing here is a statistically
  proven edge on ~13-21 months of OOS hourly data. The honest next
  steps are: more history (minute-level, 8+ years), event/session
  filters, and letting the SMC families accumulate sample size.
- Costs modeled: spread + slippage per side + commissions. Untracked:
  overnight funding on CFDs, futures roll gaps, extreme-event slippage.
