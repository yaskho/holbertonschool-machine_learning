-- Creates a view need_meeting that lists all students with a score < 80 and no last_meeting or > 1 month
CREATE VIEW need_meeting AS
SELECT name
FROM students
WHERE score < 80
  AND (last_meeting IS NULL OR last_meeting < ADDDATE(CURDATE(), INTERVAL -1 MONTH));
