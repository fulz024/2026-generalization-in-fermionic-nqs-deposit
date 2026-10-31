"""Role-based color language shared by the PRL main figures."""

# Ansatz families.  Hue encodes the coefficient map throughout the paper;
# line style and marker distinguish members within a family.
DIRECT = "#4477AA"
DIRECT_LIGHT = "#7FA6CC"
DIRECT_MID = "#5B8DB8"
MPS = "#228833"
MPS_LIGHT = "#69A96B"
NNBF = "#EE7733"
NNBF_LIGHT = "#F3A36B"
DETERMINANT_COMPONENT = "#C85C1E"
SYMMETRIC = "#AA3377"
JASTROW = "#CCBB44"
PHYSICAL_BASELINE = "#777777"
PHYSICAL_BASELINE_DARK = "#444444"

# Deprecated carrier names and auxiliary constructions remain neutral.  They
# are kept as aliases because several frozen SI scripts still import them.
SCAFFOLD = PHYSICAL_BASELINE
ALT_BACKBONE = PHYSICAL_BASELINE_DARK
NEURAL_CORRECTION = "#777777"

# References and visual structure.
REFERENCE = "#222222"
NULL = "#A0A0A0"
BOUNDARY = "#8B1E2D"
GRID = "#D8D8D8"
