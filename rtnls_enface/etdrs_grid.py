from functools import cached_property

import numpy as np
from skimage import measure


class ETDRSGrid:
    subfields_9 = "CSF", "SIM", "NIM", "TIM", "IIM", "SOM", "NOM", "TOM", "IOM"
    rings_3 = "center", "inner", "outer"
    quadrants = "superior_grid", "nasal_grid", "inferior_grid", "temporal_grid"
    all_fields = tuple([*subfields_9, *rings_3, *quadrants, "total"])

    def __init__(self, h, w, fovea_x, fovea_y, resolution, laterality):
        """
        h: height of the image
        w: width of the image
        fovea_x: x coordinate of the fovea
        fovea_y: y coordinate of the fovea
        resolution: resolution of the image in mm/pix
        laterality: laterality of the eye, 'R' or 'L'
        """
        self.h = h
        self.w = w
        self.fovea_x = fovea_x
        self.fovea_y = fovea_y

        self.resolution = resolution
        self.laterality = laterality

    def calculate_area(self, binary_image):
        return binary_image.sum() * self.resolution**2

    def calculate_count(self, binary_image):
        return np.max(measure.label(binary_image))

    def get_summary(self, binary_image, fields, include_area=True, include_count=True):
        masked_images = {field: getattr(self, field) & binary_image for field in fields}
        result = {}
        for field, masked_image in masked_images.items():
            if include_area:
                result[f"{field}_area"] = self.calculate_area(masked_image)
            if include_count:
                result[f"{field}_count"] = self.calculate_count(masked_image)

        return result

    @cached_property
    def dy(self):
        return np.arange(self.h)[:, None] - self.fovea_y

    @cached_property
    def dx(self):
        return np.arange(self.w)[None, :] - self.fovea_x

    @cached_property
    def theta(self):
        return np.arctan2(self.dy, self.dx) / (2 * np.pi)

    @cached_property
    def distance_to_fovea(self):
        return self.resolution * np.sqrt(self.dx * self.dx + self.dy * self.dy)

    @cached_property
    def total(self):
        return np.ones((self.h, self.w), dtype=bool)

    """rings"""

    @cached_property
    def center(self):
        return self.distance_to_fovea < 1

    @cached_property
    def inner(self):
        return (self.distance_to_fovea < 3) & ~self.center

    @cached_property
    def outer(self):
        return (self.distance_to_fovea < 6) & ~(self.center | self.inner)

    @cached_property
    def grid(self):
        return self.center | self.inner | self.outer

    """quadrants"""

    @cached_property
    def inferior(self):
        return (1 / 8 < self.theta) & (self.theta <= 3 / 8)

    @cached_property
    def left(self):
        return (3 / 8 < self.theta) | (self.theta <= -3 / 8)

    @cached_property
    def superior(self):
        return (-3 / 8 < self.theta) & (self.theta <= -1 / 8)

    @cached_property
    def right(self):
        return (-1 / 8 < self.theta) & (self.theta <= 1 / 8)

    @cached_property
    def nasal(self):
        return self.right if self.laterality == "R" else self.left

    @cached_property
    def temporal(self):
        return self.left if self.laterality == "R" else self.right

    """quadrants grid"""

    @cached_property
    def superior_grid(self):
        return self.superior & self.grid

    @cached_property
    def nasal_grid(self):
        return self.nasal & self.grid

    @cached_property
    def inferior_grid(self):
        return self.inferior & self.grid

    @cached_property
    def temporal_grid(self):
        return self.temporal & self.grid

    """subfields"""

    @cached_property
    def CSF(self):
        return self.center

    @cached_property
    def SIM(self):
        return self.superior & self.inner

    @cached_property
    def NIM(self):
        return self.nasal & self.inner

    @cached_property
    def TIM(self):
        return self.temporal & self.inner

    @cached_property
    def IIM(self):
        return self.inferior & self.inner

    @cached_property
    def NOM(self):
        return self.nasal & self.outer

    @cached_property
    def TOM(self):
        return self.temporal & self.outer

    @cached_property
    def SOM(self):
        return self.superior & self.outer

    @cached_property
    def IOM(self):
        return self.inferior & self.outer

    def create_svg(self, text_dict):
        d_inner = np.sqrt((5**2) / 2)
        d_outer = np.sqrt((30**2) / 2)

        stroke_width = 0.2
        if self.laterality == "R":
            # nasal right, temporal left
            translate = str.maketrans({"N": "R", "T": "L"})
        else:
            # nasal left, temporal right
            translate = str.maketrans({"T": "R", "N": "L"})
        text_query = {k.translate(translate): v for k, v in text_dict.items()}

        def svg_element(element, **kwargs):
            attr = " ".join(f'{k}="{v}"' for k, v in kwargs.items())
            return f"<{element} {attr} />"

        def svg_text(x, y, text):
            return f'<text x="{x}" y="{y}" fill="black" font-family="Verdana" font-size="4" text-anchor="middle" dominant-baseline="middle">{text}</text>'

        text_coordinates = [
            (30, 30, text_query.get("CSF", "CSF")),
            (30, 40, text_query.get("IIM", "IIM")),
            (30, 20, text_query.get("SIM", "SIM")),
            (20, 30, text_query.get("LIM", "LIM")),
            (40, 30, text_query.get("RIM", "RIM")),
            (30, 52.5, text_query.get("IOM", "IOM")),
            (30, 7.5, text_query.get("SOM", "SOM")),
            (7.5, 30, text_query.get("LOM", "LOM")),
            (52.5, 30, text_query.get("ROM", "ROM")),
        ]

        svg_texts = "\n".join(svg_text(x, y, text) for (x, y, text) in text_coordinates)

        svg_data = f"""
        <svg width="240" height="240" viewBox="0 0 {2*30} {2*30}" xmlns="http://www.w3.org/2000/svg">
            <style>
                circle, line {{
                    stroke-width: {stroke_width};
                }}
            </style>
            {svg_element('circle', cx=30, cy=30, r=30, stroke="black", fill="none")}
            {svg_element('circle', cx=30, cy=30, r=15, stroke="black", fill="none")}
            {svg_element('circle', cx=30, cy=30, r=5, stroke="black", fill="none")}
            {svg_element('line', x1=30-d_inner, y1=30-d_inner, x2=30-d_outer, y2=30-d_outer, stroke="black")}
            {svg_element('line', x1=30+d_inner, y1=30-d_inner, x2=30+d_outer, y2=30-d_outer, stroke="black")}
            {svg_element('line', x1=30-d_inner, y1=30+d_inner, x2=30-d_outer, y2=30+d_outer, stroke="black")}
            {svg_element('line', x1=30+d_inner, y1=30+d_inner, x2=30+d_outer, y2=30+d_outer, stroke="black")}
            
            {svg_texts}
        
        </svg>
        """

        return svg_data
