/* ##########################
Training Instances (First Pass)

CS 230
Sal Calvo
Adam Gotlin
Mike Hittle

This script populates a new SQL table called "cs230_training_instances", which contains a re-arranged version of the
"time_series_all" table Jessica and Rachel built. Where each row of that table contains the Lumo accelerometer metric
and the GPS metrics, this table's rows contain the following:

(1) the feature to predict (gps_speed at an instant)
(2) the key accelerometer features at that instant (cadence and bounce, for example)
(3) those accelerometer features with a lag of "n" events ago

############################# */

WITH timeseries AS (SELECT ROW_NUMBER() OVER (PARTITION BY activity_id) as row_number,
  user_id,
  activity_id,
  cadence,
  bounce,
  gps_speed
    FROM timeseries_all_01122017
WHERE activity_id IN (7532677280941, 6582421841841)
ORDER BY metric_utc asc)

SELECT
  lag_0.activity_id,
  lag_0.row_number,
  p.gender,
  p.age,
  p.height,
  p.weight,
  lag_5.cadence as cadence_lag_5,
  lag_4.cadence as cadence_lag_4,
  lag_3.cadence as cadence_lag_3,
  lag_2.cadence as cadence_lag_2,
  lag_1.cadence as cadence_lag_1,
  lag_0.cadence as cadence_lag_0,
  lag_5.bounce as bounce_lag_5,
  lag_4.bounce as bounce_lag_4,
  lag_3.bounce as bounce_lag_3,
  lag_2.bounce as bounce_lag_2,
  lag_1.bounce as bounce_lag_1,
  lag_0.bounce as bounce_lag_0,
  lag_0.gps_speed
    FROM timeseries lag_0
      LEFT JOIN profile_01122017 p ON lag_0.user_id = p.user_id
      LEFT JOIN timeseries lag_1 ON lag_0.activity_id = lag_1.activity_id AND
                                    lag_0.row_number - 1 = lag_1.row_number
      LEFT JOIN timeseries lag_2 ON lag_0.activity_id = lag_2.activity_id AND
                                    lag_0.row_number - 2 = lag_2.row_number
      LEFT JOIN timeseries lag_3 ON lag_0.activity_id = lag_3.activity_id AND
                                    lag_0.row_number - 3 = lag_3.row_number
      LEFT JOIN timeseries lag_4 ON lag_0.activity_id = lag_4.activity_id AND
                                    lag_0.row_number - 4 = lag_4.row_number
      LEFT JOIN timeseries lag_5 ON lag_0.activity_id = lag_5.activity_id AND
                                    lag_0.row_number - 5 = lag_5.row_number
ORDER BY lag_0.activity_id, row_number asc


SELECT user_id, COUNT(DISTINCT activity_id)
from timeseries_all_01122017
GROUP BY 1
ORDER BY 2 DESC
LIMIT 100