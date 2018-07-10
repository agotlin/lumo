/* ##########################
Finding Good Run Instances

CS 230
Sal Calvo
Adam Gotlin
Mike Hittle

########################## */

SELECT activity_id, COUNT(*)
FROM timeseries_all_01122017
GROUP BY 1
ORDER BY 2 DESC
LIMIT 10000

-- A nice run someone went on, March 19 2017, from 22:03 UTC to March 00:09 UTC,
-- a ~2hr period.

WITH timeseries AS
SELECT *, ROW_NUMBER() OVER (PARTITION BY metric_seconds) as row_number FROM timeseries_all_01122017
WHERE activity_id = 5679257960941
ORDER BY metric_utc DESC
LIMIT 100
