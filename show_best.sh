command ls -1 cv | \
awk -F'_' \
'BEGIN { best = 9999 } \
{ tmp = substr($4, 1, 6); if (best > substr($4, 1, 6)) { best = tmp }} \
END { print best }' \
