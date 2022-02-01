data = readtable('20190101_520813766.csv');
data = data(12:40,:);
origin = table2array(data(1, {'latitude', 'longitude'}));
m2ft = 3.281;
ellipsoid = [6378.137 0.0818191910428158]*m2ft;

function [distance] = haversine(point,origin)
    lat1 = origin(1)
    lon1 = origin(2)
    lat1 = point(1)
    lon1 = point(2)
    
    
    
    
end