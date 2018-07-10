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

-- The below uses all usable (10 continuous non-null sequenced) data for the user_id
-- in the very first WHERE statement.

WITH timeseries AS (SELECT ROW_NUMBER() OVER (PARTITION BY metric_seconds) as deduplicator,
  activity_id,
  user_id,
  metric_utc,
  cadence,
  bounce,
  braking,
  ground_contact,
  pelvic_drop,
  pelvic_rotation,
  pelvic_tilt,
  gps_speed
    FROM timeseries_all_01122017
WHERE user_id IN ('9366289f-e52d-42f2-8558-e01e87b8c13a')
ORDER BY metric_utc asc),

timeseriescleansed AS (SELECT ROW_NUMBER() OVER (PARTITION BY activity_id) as row_number,
  activity_id,
  user_id,
  metric_utc,
  cadence,
  bounce,
  braking,
  ground_contact,
  pelvic_drop,
  pelvic_rotation,
  pelvic_tilt,
  gps_speed
FROM timeseries
    WHERE deduplicator = 1),

activity_ledger AS (SELECT
  lag_0.activity_id,
  lag_0.user_id,
  lag_0.row_number,
  p.gender,
  p.age,
  p.height,
  p.weight,
  lag_10.bounce as bounce_lag_10,
  lag_9.bounce as bounce_lag_9,
  lag_8.bounce as bounce_lag_8,
  lag_7.bounce as bounce_lag_7,
  lag_6.bounce as bounce_lag_6,
  lag_5.bounce as bounce_lag_5,
  lag_4.bounce as bounce_lag_4,
  lag_3.bounce as bounce_lag_3,
  lag_2.bounce as bounce_lag_2,
  lag_1.bounce as bounce_lag_1,
  lag_0.bounce as bounce_lag_0,
  lag_10.braking as braking_lag_10,
  lag_9.braking as braking_lag_9,
  lag_8.braking as braking_lag_8,
  lag_7.braking as braking_lag_7,
  lag_6.braking as braking_lag_6,
  lag_5.braking as braking_lag_5,
  lag_4.braking as braking_lag_4,
  lag_3.braking as braking_lag_3,
  lag_2.braking as braking_lag_2,
  lag_1.braking as braking_lag_1,
  lag_0.braking as braking_lag_0,
  lag_10.cadence as cadence_lag_10,
  lag_9.cadence as cadence_lag_9,
  lag_8.cadence as cadence_lag_8,
  lag_7.cadence as cadence_lag_7,
  lag_6.cadence as cadence_lag_6,
  lag_5.cadence as cadence_lag_5,
  lag_4.cadence as cadence_lag_4,
  lag_3.cadence as cadence_lag_3,
  lag_2.cadence as cadence_lag_2,
  lag_1.cadence as cadence_lag_1,
  lag_0.cadence as cadence_lag_0,
  lag_10.ground_contact as ground_contact_lag_10,
  lag_9.ground_contact as ground_contact_lag_9,
  lag_8.ground_contact as ground_contact_lag_8,
  lag_7.ground_contact as ground_contact_lag_7,
  lag_6.ground_contact as ground_contact_lag_6,
  lag_5.ground_contact as ground_contact_lag_5,
  lag_4.ground_contact as ground_contact_lag_4,
  lag_3.ground_contact as ground_contact_lag_3,
  lag_2.ground_contact as ground_contact_lag_2,
  lag_1.ground_contact as ground_contact_lag_1,
  lag_0.ground_contact as ground_contact_lag_0,
  lag_10.pelvic_drop as pelvic_drop_lag_10,
  lag_9.pelvic_drop as pelvic_drop_lag_9,
  lag_8.pelvic_drop as pelvic_drop_lag_8,
  lag_7.pelvic_drop as pelvic_drop_lag_7,
  lag_6.pelvic_drop as pelvic_drop_lag_6,
  lag_5.pelvic_drop as pelvic_drop_lag_5,
  lag_4.pelvic_drop as pelvic_drop_lag_4,
  lag_3.pelvic_drop as pelvic_drop_lag_3,
  lag_2.pelvic_drop as pelvic_drop_lag_2,
  lag_1.pelvic_drop as pelvic_drop_lag_1,
  lag_0.pelvic_drop as pelvic_drop_lag_0,
  lag_10.pelvic_rotation as pelvic_rotation_lag_10,
  lag_9.pelvic_rotation as pelvic_rotation_lag_9,
  lag_8.pelvic_rotation as pelvic_rotation_lag_8,
  lag_7.pelvic_rotation as pelvic_rotation_lag_7,
  lag_6.pelvic_rotation as pelvic_rotation_lag_6,
  lag_5.pelvic_rotation as pelvic_rotation_lag_5,
  lag_4.pelvic_rotation as pelvic_rotation_lag_4,
  lag_3.pelvic_rotation as pelvic_rotation_lag_3,
  lag_2.pelvic_rotation as pelvic_rotation_lag_2,
  lag_1.pelvic_rotation as pelvic_rotation_lag_1,
  lag_0.pelvic_rotation as pelvic_rotation_lag_0,
  lag_10.pelvic_tilt as pelvic_rotation_lag_10,
  lag_9.pelvic_tilt as pelvic_tilt_lag_9,
  lag_8.pelvic_tilt as pelvic_tilt_lag_8,
  lag_7.pelvic_tilt as pelvic_tilt_lag_7,
  lag_6.pelvic_tilt as pelvic_tilt_lag_6,
  lag_5.pelvic_tilt as pelvic_tilt_lag_5,
  lag_4.pelvic_tilt as pelvic_tilt_lag_4,
  lag_3.pelvic_tilt as pelvic_tilt_lag_3,
  lag_2.pelvic_tilt as pelvic_tilt_lag_2,
  lag_1.pelvic_tilt as pelvic_tilt_lag_1,
  lag_0.pelvic_tilt as pelvic_tilt_lag_0,
  lag_5.gps_speed as speed_lag_5
    FROM timeseriescleansed lag_0
      LEFT JOIN profile_01122017 p ON lag_0.user_id = p.user_id
      LEFT JOIN timeseriescleansed lag_1 ON lag_0.activity_id = lag_1.activity_id AND
                                    lag_0.row_number - 1 = lag_1.row_number
      LEFT JOIN timeseriescleansed lag_2 ON lag_0.activity_id = lag_2.activity_id AND
                                    lag_0.row_number - 2 = lag_2.row_number
      LEFT JOIN timeseriescleansed lag_3 ON lag_0.activity_id = lag_3.activity_id AND
                                    lag_0.row_number - 3 = lag_3.row_number
      LEFT JOIN timeseriescleansed lag_4 ON lag_0.activity_id = lag_4.activity_id AND
                                    lag_0.row_number - 4 = lag_4.row_number
      LEFT JOIN timeseriescleansed lag_5 ON lag_0.activity_id = lag_5.activity_id AND
                                    lag_0.row_number - 5 = lag_5.row_number
      LEFT JOIN timeseriescleansed lag_6 ON lag_0.activity_id = lag_6.activity_id AND
                                    lag_0.row_number - 6 = lag_6.row_number
      LEFT JOIN timeseriescleansed lag_7 ON lag_0.activity_id = lag_7.activity_id AND
                                    lag_0.row_number - 7 = lag_7.row_number
      LEFT JOIN timeseriescleansed lag_8 ON lag_0.activity_id = lag_8.activity_id AND
                                    lag_0.row_number - 8 = lag_8.row_number
      LEFT JOIN timeseriescleansed lag_9 ON lag_0.activity_id = lag_9.activity_id AND
                                    lag_0.row_number - 9 = lag_9.row_number
      LEFT JOIN timeseriescleansed lag_10 ON lag_0.activity_id = lag_10.activity_id AND
                                    lag_0.row_number - 10 = lag_10.row_number
ORDER BY lag_0.activity_id, row_number asc)

SELECT * FROM activity_ledger WHERE (activity_ledger IS NOT NULL)

-- The below is the same thing without the lags:

WITH timeseries AS (SELECT ROW_NUMBER() OVER (PARTITION BY activity_id, metric_seconds) as deduplicator,
  activity_id,
  user_id,
  metric_utc,
  cadence,
  bounce,
  braking,
  ground_contact,
  pelvic_drop,
  pelvic_rotation,
  pelvic_tilt,
  gps_speed
    FROM timeseries_all_01122017
WHERE activity_id IN (SELECT activity_id FROM

                        (SELECT activity_id, COUNT(*)
                      FROM timeseries_all_01122017
                      GROUP BY 1
                      ORDER BY 2 DESC
                      LIMIT 1000) biggest_1000_runs
)
ORDER BY metric_utc asc),

timeseriescleansed AS (SELECT ROW_NUMBER() OVER (PARTITION BY activity_id) as row_number,
  activity_id,
  user_id,
  metric_utc,
  cadence,
  bounce,
  braking,
  ground_contact,
  pelvic_drop,
  pelvic_rotation,
  pelvic_tilt,
  gps_speed as gps_speed_true
FROM timeseries
    WHERE deduplicator = 1),

activity_ledger AS (SELECT
  lag_0.activity_id,
  lag_0.user_id,
  date_part('epoch', lag_0.metric_utc) as metric_utc_unix,
  p.gender,
  p.age,
  p.height,
  p.weight,
  lag_0.bounce as bounce,
  lag_0.braking as braking,
  lag_0.cadence as cadence,
  lag_0.ground_contact as ground_contact,
  lag_0.pelvic_drop as pelvic_drop,
  lag_0.pelvic_rotation as pelvic_rotation,
  lag_0.pelvic_tilt as pelvic_tilt,
  lag_0.gps_speed_true as gps_speed_true
    FROM timeseriescleansed lag_0
      LEFT JOIN profile_01122017 p ON lag_0.user_id = p.user_id
ORDER BY lag_0.activity_id, row_number asc)

SELECT *,
round(gps_speed_true) as gps_speed_unit,
round(round(2 * gps_speed_true)/2, 1) as gps_speed_half_unit,
round(gps_speed_true, 1) as gps_speed_unit_1dec,
round(gps_speed_true, 2) as gps_speed_unit_2dec
FROM activity_ledger WHERE (activity_ledger IS NOT NULL)