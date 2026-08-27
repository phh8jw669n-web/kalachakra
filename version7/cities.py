"""A curated, structured global dataset of major metropolitan hubs.

Each entry is ``(name, latitude_deg, longitude_deg, region)``. This is the "high-density
global dataset of major urban hubs" the version7 trainer samples (alongside a regional
lat/lon grid) so the learned colour field is dense and accurate over the territories people
actually live in. The same list is exported to ``cities.json`` for the frontend, which draws
each hub as an interactive marker on the globe.
"""

from __future__ import annotations

#: (name, lat, lon, region) — ~130 of the world's largest / most significant metros.
CITIES: list[tuple[str, float, float, str]] = [
    # --- Asia ---
    ("Tokyo", 35.6895, 139.6917, "Asia"),
    ("Delhi", 28.6139, 77.2090, "Asia"),
    ("Shanghai", 31.2304, 121.4737, "Asia"),
    ("Mumbai", 19.0760, 72.8777, "Asia"),
    ("Beijing", 39.9042, 116.4074, "Asia"),
    ("Dhaka", 23.8103, 90.4125, "Asia"),
    ("Osaka", 34.6937, 135.5023, "Asia"),
    ("Karachi", 24.8607, 67.0011, "Asia"),
    ("Chongqing", 29.4316, 106.9123, "Asia"),
    ("Istanbul", 41.0082, 28.9784, "Asia"),
    ("Kolkata", 22.5726, 88.3639, "Asia"),
    ("Manila", 14.5995, 120.9842, "Asia"),
    ("Guangzhou", 23.1291, 113.2644, "Asia"),
    ("Shenzhen", 22.5431, 114.0579, "Asia"),
    ("Bangalore", 12.9716, 77.5946, "Asia"),
    ("Jakarta", -6.2088, 106.8456, "Asia"),
    ("Seoul", 37.5665, 126.9780, "Asia"),
    ("Bangkok", 13.7563, 100.5018, "Asia"),
    ("Chennai", 13.0827, 80.2707, "Asia"),
    ("Hyderabad", 17.3850, 78.4867, "Asia"),
    ("Ho Chi Minh City", 10.8231, 106.6297, "Asia"),
    ("Hong Kong", 22.3193, 114.1694, "Asia"),
    ("Hanoi", 21.0278, 105.8342, "Asia"),
    ("Tehran", 35.6892, 51.3890, "Asia"),
    ("Baghdad", 33.3152, 44.3661, "Asia"),
    ("Riyadh", 24.7136, 46.6753, "Asia"),
    ("Singapore", 1.3521, 103.8198, "Asia"),
    ("Lahore", 31.5204, 74.3587, "Asia"),
    ("Kuala Lumpur", 3.1390, 101.6869, "Asia"),
    ("Taipei", 25.0330, 121.5654, "Asia"),
    ("Ahmedabad", 23.0225, 72.5714, "Asia"),
    ("Pune", 18.5204, 73.8567, "Asia"),
    ("Chengdu", 30.5728, 104.0668, "Asia"),
    ("Xi'an", 34.3416, 108.9398, "Asia"),
    ("Dubai", 25.2048, 55.2708, "Asia"),
    ("Yangon", 16.8409, 96.1735, "Asia"),
    ("Jeddah", 21.4858, 39.1925, "Asia"),
    ("Almaty", 43.2220, 76.8512, "Asia"),
    ("Colombo", 6.9271, 79.8612, "Asia"),
    ("Kathmandu", 27.7172, 85.3240, "Asia"),
    # --- Europe ---
    ("Moscow", 55.7558, 37.6173, "Europe"),
    ("London", 51.5074, -0.1278, "Europe"),
    ("Paris", 48.8566, 2.3522, "Europe"),
    ("Madrid", 40.4168, -3.7038, "Europe"),
    ("Barcelona", 41.3874, 2.1686, "Europe"),
    ("Berlin", 52.5200, 13.4050, "Europe"),
    ("Rome", 41.9028, 12.4964, "Europe"),
    ("Kyiv", 50.4501, 30.5234, "Europe"),
    ("Saint Petersburg", 59.9311, 30.3609, "Europe"),
    ("Milan", 45.4642, 9.1900, "Europe"),
    ("Vienna", 48.2082, 16.3738, "Europe"),
    ("Warsaw", 52.2297, 21.0122, "Europe"),
    ("Amsterdam", 52.3676, 4.9041, "Europe"),
    ("Athens", 37.9838, 23.7275, "Europe"),
    ("Lisbon", 38.7223, -9.1393, "Europe"),
    ("Stockholm", 59.3293, 18.0686, "Europe"),
    ("Munich", 48.1351, 11.5820, "Europe"),
    ("Brussels", 50.8503, 4.3517, "Europe"),
    ("Copenhagen", 55.6761, 12.5683, "Europe"),
    ("Dublin", 53.3498, -6.2603, "Europe"),
    ("Prague", 50.0755, 14.4378, "Europe"),
    ("Budapest", 47.4979, 19.0402, "Europe"),
    ("Zurich", 47.3769, 8.5417, "Europe"),
    ("Helsinki", 60.1699, 24.9384, "Europe"),
    ("Oslo", 59.9139, 10.7522, "Europe"),
    # --- Africa ---
    ("Lagos", 6.5244, 3.3792, "Africa"),
    ("Cairo", 30.0444, 31.2357, "Africa"),
    ("Kinshasa", -4.4419, 15.2663, "Africa"),
    ("Johannesburg", -26.2041, 28.0473, "Africa"),
    ("Cape Town", -33.9249, 18.4241, "Africa"),
    ("Nairobi", -1.2921, 36.8219, "Africa"),
    ("Casablanca", 33.5731, -7.5898, "Africa"),
    ("Addis Ababa", 9.0300, 38.7400, "Africa"),
    ("Accra", 5.6037, -0.1870, "Africa"),
    ("Dar es Salaam", -6.7924, 39.2083, "Africa"),
    ("Khartoum", 15.5007, 32.5599, "Africa"),
    ("Algiers", 36.7538, 3.0588, "Africa"),
    ("Luanda", -8.8390, 13.2894, "Africa"),
    ("Abidjan", 5.3600, -4.0083, "Africa"),
    ("Tunis", 36.8065, 10.1815, "Africa"),
    ("Dakar", 14.7167, -17.4677, "Africa"),
    ("Kampala", 0.3476, 32.5825, "Africa"),
    # --- North America ---
    ("Mexico City", 19.4326, -99.1332, "North America"),
    ("New York", 40.7128, -74.0060, "North America"),
    ("Los Angeles", 34.0522, -118.2437, "North America"),
    ("Chicago", 41.8781, -87.6298, "North America"),
    ("Toronto", 43.6532, -79.3832, "North America"),
    ("Houston", 29.7604, -95.3698, "North America"),
    ("Guadalajara", 20.6597, -103.3496, "North America"),
    ("Montreal", 45.5017, -73.5673, "North America"),
    ("Miami", 25.7617, -80.1918, "North America"),
    ("San Francisco", 37.7749, -122.4194, "North America"),
    ("Vancouver", 49.2827, -123.1207, "North America"),
    ("Washington", 38.9072, -77.0369, "North America"),
    ("Havana", 23.1136, -82.3666, "North America"),
    ("Guatemala City", 14.6349, -90.5069, "North America"),
    ("Seattle", 47.6062, -122.3321, "North America"),
    ("Atlanta", 33.7490, -84.3880, "North America"),
    ("Dallas", 32.7767, -96.7970, "North America"),
    # --- South America ---
    ("Sao Paulo", -23.5505, -46.6333, "South America"),
    ("Buenos Aires", -34.6037, -58.3816, "South America"),
    ("Rio de Janeiro", -22.9068, -43.1729, "South America"),
    ("Lima", -12.0464, -77.0428, "South America"),
    ("Bogota", 4.7110, -74.0721, "South America"),
    ("Santiago", -33.4489, -70.6693, "South America"),
    ("Caracas", 10.4806, -66.9036, "South America"),
    ("Brasilia", -15.7939, -47.8828, "South America"),
    ("Medellin", 6.2476, -75.5658, "South America"),
    ("Quito", -0.1807, -78.4678, "South America"),
    ("Montevideo", -34.9011, -56.1645, "South America"),
    ("La Paz", -16.4897, -68.1193, "South America"),
    # --- Oceania ---
    ("Sydney", -33.8688, 151.2093, "Oceania"),
    ("Melbourne", -37.8136, 144.9631, "Oceania"),
    ("Brisbane", -27.4698, 153.0251, "Oceania"),
    ("Perth", -31.9505, 115.8605, "Oceania"),
    ("Auckland", -36.8485, 174.7633, "Oceania"),
    ("Wellington", -41.2865, 174.7762, "Oceania"),
    ("Honolulu", 21.3069, -157.8583, "Oceania"),
    ("Suva", -18.1248, 178.4501, "Oceania"),
    ("Port Moresby", -9.4438, 147.1803, "Oceania"),
    # --- high-latitude anchors (keep the field honest near the poles) ---
    ("Reykjavik", 64.1466, -21.9426, "Europe"),
    ("Anchorage", 61.2181, -149.9003, "North America"),
    ("Murmansk", 68.9585, 33.0827, "Europe"),
    ("Ushuaia", -54.8019, -68.3030, "South America"),
    ("Longyearbyen", 78.2232, 15.6267, "Europe"),
    ("McMurdo", -77.8419, 166.6863, "Antarctica"),
]


def unique_cities() -> list[tuple[str, float, float, str]]:
    """The city list with any duplicate names removed (keeps first occurrence)."""
    seen: set[str] = set()
    out: list[tuple[str, float, float, str]] = []
    for row in CITIES:
        if row[0] in seen:
            continue
        seen.add(row[0])
        out.append(row)
    return out
