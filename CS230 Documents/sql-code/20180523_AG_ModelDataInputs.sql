/* ##########################
Training Instances

CS 230
Sal Calvo
Adam Gotlin
Mike Hittle

This script populates various SQL tables for inputs to our models.


############################# */

/* ##########################

RAW ALTER TABLES

select * from timeseries_all_01122017 limit 100;
select * from profile_01122017 limit 100;

############################# */

/* ##########################

  DATA INVESTIGATIONS

  ############################# */

-- Create a list of longest 1,000 runs

  SELECT activity_id, COUNT(*)
    INTO list_of_biggest_1000_runs_AG_20180523
    FROM timeseries_all_01122017
                      GROUP BY 1
                      ORDER BY 2 DESC
                      LIMIT 1000) biggest_1000_runs

  SELECT * FROM list_of_biggest_1000_runs_AG_20180523

-- Check the duplicates in the top 1,000 run timeseries table

  select activity_id , metric_utc
    from timeseries_TOP_1000_RUNS
      group by activity_id , metric_utc
        having count(distinct cadence) > 1
          or count(distinct bounce) > 1
          or count(distinct braking) > 1
          or count(distinct ground_contact) > 1
          or count(distinct pelvic_drop) > 1
          or count(distinct pelvic_rotation) > 1
          or count(distinct pelvic_tilt) > 1
          or count(distinct gps_speed) > 1
          -- or count(distinct deduplicator) > 4 -- for whatever reason, there are ~4 duplicates per row

-- Check there are only two genders
  SELECT height, count(*)  -- change column of interest
    FROM profile_01122017
    GROUP BY height
    ORDER BY height
    /* 	NULL 43
        m	5
        male	6359
        female	4460 */

-- Figure out gender case when DELETE
    select
      (case p.gender when 'male' then 1 when 'female' then -1 else 0 end) as gender , -- as p.genderNum,
      p.gender,
      p.age
      from profile_01122017 p
      limit 1000;

/* ##########################

  TIME SERIES CNN

  File contains
  (1) the label to predict (gps_speed at each timestamp instant)
    There are multiple "versions" of the labeling (i.e. rounding to nearest unit, half unit, etc.)
  (2) the key accelerometer features at that instant (cadence and bounce, for example)
  (3) anthropometric data for the runnerID

  ############################# */

WITH timeseries AS (
  SELECT
    ROW_NUMBER() OVER (PARTITION BY timeseries_all_01122017.activity_id, metric_seconds) as deduplicator,
    timeseries_all_01122017.activity_id,
    user_id,
    metric_utc,
    metric_seconds, -- added
    cadence,
    bounce,
    braking,
    ground_contact,
    pelvic_drop,
    pelvic_rotation,
    pelvic_tilt,
    gps_speed
    -- into TEMPORARY TABLE timeseries_TOP_1000_RUNS -- run when investigating this table for errors
    FROM timeseries_all_01122017
      inner join list_of_biggest_1000_runs_AG_20180523
          on (timeseries_all_01122017.activity_id = list_of_biggest_1000_runs_AG_20180523.activity_id)
  ORDER BY activity_id, metric_utc asc ), -- This is changed to activity_id, metric_utc

-- Remove duplicates (NOTE: CONFIRM DUPLICATE ARE NOT UNIQUE BEFORE USING (see above))
timeseriescleansed AS (SELECT ROW_NUMBER() OVER (PARTITION BY activity_id) as row_number,
  activity_id,
  user_id,
  metric_utc, -- drop the metric_seconds which is redundant with metric_utc
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
  (case p.gender when 'male' then 1 when 'female' then -1 else 0 end) as gender ,
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
  lag_0.gps_speed_true as gps_speed_true,
  round(gps_speed_true) as gps_speed_round_unit,
  round(round(2 * gps_speed_true)/2, 1) as gps_speed_round_half_unit,
  round(gps_speed_true, 1) as gps_speed_round_1dec,
  round(gps_speed_true, 2) as gps_speed_round_2dec
    FROM timeseriescleansed lag_0
      LEFT JOIN profile_01122017 p ON lag_0.user_id = p.user_id
    WHERE p.gender in ('male', 'female')
ORDER BY lag_0.activity_id, lag_0.metric_utc, row_number asc)

  SELECT *
    -- INTO TimeSeries_Longest1000Runs_20180523
    FROM activity_ledger
    WHERE activity_ledger IS NOT NULL -- ;
    LIMIT 10000; -- include to produce a small sample for easy viewing

/* COPY activity_ledger_noNull
       TO 'C:\Users\adam\Documents\CS 230\Project' DELIMITER ',' CSV HEADER; */
  -- remove any rows with NULL

/*

Scratch

alternative where clause for top 1000 runs
/* WHERE activity_id IN (SELECT activity_id FROM
                        (SELECT activity_id, COUNT(*)
                      FROM timeseries_all_01122017
                      GROUP BY 1
                      ORDER BY 2 DESC
                      LIMIT 1000) biggest_1000_runs */

alternative way to query the final acitivty_ledger table
,
    round(gps_speed_true) as gps_speed_current,
    round(round(2 * gps_speed_true)/2, 1) as gps_speed_half_unit,
    round(gps_speed_true, 1) as gps_speed_unit_1dec,
    round(gps_speed_true, 2) as gps_speed_unit_2dec

 */



/* ##########################

  Vectorized Input

  File contains
  (1) the label to predict (gps_speed at each timestamp instant)
    There are multiple "versions" of the labeling (i.e. rounding to nearest unit, half unit, etc., lag_0, lag_5, etc.)
  (2) the key accelerometer features at that instant (cadence and bounce, for example)
  (3) anthropometric data for the runnerID

  ############################# */

-- The below uses all usable (10 continuous non-null sequenced) data for the user_id
-- in the very first WHERE statement.

WITH timeseries AS (
  SELECT
  ROW_NUMBER() OVER (PARTITION BY t.activity_id, metric_seconds) as deduplicator,
  t.activity_id,
  user_id,
  metric_utc,
  metric_seconds,
  cadence,
  bounce,
  braking,
  ground_contact,
  pelvic_drop,
  pelvic_rotation,
  pelvic_tilt,
  gps_speed
  FROM timeseries_all_01122017 t
    INNER JOIN list_of_biggest_1000_runs_AG_20180523 l
          on (t.activity_id = l.activity_id) -- This is c
  --WHERE user_id IN ('9366289f-e52d-42f2-8558-e01e87b8c13a')
ORDER BY t.activity_id, t.metric_utc asc
),

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
  date_part('epoch', lag_0.metric_utc) as metric_utc_unix_lag_0,
  lag_0.row_number as row_Number,
  (case p.gender when 'male' then 1 when 'female' then -1 else 0 end) as gender ,
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
  lag_10.pelvic_tilt as pelvic_tilt_lag_10,
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
  lag_0.gps_speed as gps_speed_lag_0,
  lag_5.gps_speed as gps_speed_lag_5,
  --round(round(2 * lag_0.gps_speed)/2, 1) as gps_speed_round_half_unit_lag_0,
  round(lag_0.gps_speed, 1) as gps_speed_round_1dec_lag_0,
  --round(lag_0.gps_speed, 2) as gps_speed_round_2dec_lag_0,
  --round(round(2 * lag_5.gps_speed)/2, 1) as gps_speed_round_half_unit_lag_5,
  round(lag_5.gps_speed, 1) as gps_speed_round_1dec_lag_5,
  --round(lag_5.gps_speed, 2) as gps_speed_round_2dec_lag_5,
  (lag_0.gps_speed + lag_1.gps_speed + lag_2.gps_speed + lag_3.gps_speed +
      lag_4.gps_speed + lag_5.gps_speed + lag_6.gps_speed + lag_7.gps_speed +
      lag_8.gps_speed + lag_9.gps_speed + lag_10.gps_speed)/11 as gps_speed_AVG,
  round((lag_0.gps_speed + lag_1.gps_speed + lag_2.gps_speed + lag_3.gps_speed +
    lag_4.gps_speed + lag_5.gps_speed + lag_6.gps_speed + lag_7.gps_speed +
    lag_8.gps_speed + lag_9.gps_speed + lag_10.gps_speed)/11,1) as gps_speed_AVG_round_1dec
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
      WHERE p.gender in ('male', 'female')
ORDER BY lag_0.activity_id, row_number asc)

SELECT *
    INTO TimeSeries_Vector_20180523_100Rows
    FROM activity_ledger
    WHERE activity_ledger IS NOT NULL -- ;
    LIMIT 100;