import xml.etree.ElementTree as ET

from jsonargparse import CLI


def normalize_traces(inkml_path: str, fac: float = 1, step=1):
    doc_namespace = "{http://www.w3.org/2003/InkML}"
    element_tree = ET.parse(inkml_path, parser=ET.XMLParser(encoding='iso-8859-5'))

    trace_tags = element_tree.findall(doc_namespace + "trace")
    traces = []

    for tag in trace_tags:
        coordinates = [coordinate for coordinate in tag.text.split(",")]
        coordinates = [coordinate.replace("\n","").strip() for coordinate in coordinates]
        coordinates = [coordinate.split(" ") for coordinate in coordinates]
        coordinates = [(float(coordinate[0]), float(coordinate[1])) for coordinate in coordinates]

        traces.append(coordinates)

    x_coords = [coordinate[0] for coords in traces for coordinate in coords]
    y_coords = [coordinate[1] for coords in traces for coordinate in coords]


    x_min = min(x_coords)
    x_max = max(x_coords)
    y_min = min(y_coords)
    y_max = max(y_coords)

    normalized_traces = []

    for trace in traces:
        normalized_trace = [normalize(coordinate, x_min, x_max, y_min, y_max, fac=fac) for coordinate in trace[::step]]
        normalized_traces.append(normalized_trace)

    return normalized_traces

def normalize(coordinate, x_min, x_max, y_min, y_max, fac = 1):
    #delta_x = x_max - x_min
    delta_y = y_max - y_min

    normalized_x = (coordinate[0] - x_min) / delta_y
    normalized_y = (coordinate[1] - y_min) / delta_y

    return (round(normalized_x, 2) * fac, (y_max - round(normalized_y, 2)) * fac)


def main(p: str):
    traces = normalize_traces(p, fac=2.2, step=2)
    out = []

    palette = ["EF476F", "FFD166", '06D6A0', '118AB2', '073B4C']
    # create color palette once, copy to TeX file
    # for i, p in enumerate(palette):
    #     out.append("\\definecolor{palletec" + i.__str__() + "}{HTML}{" + p + "}")

    out.append("\\begin{tikzpicture}")


    for i, trace in enumerate(traces):
        col = palette[i % len(palette)]
        coords_as_strings = [(str(coord)) for coord in trace]
        out.append(f"\draw[color=palletec{i % len(palette)}, line width=1.5] " + '--'.join(coords_as_strings) + ";")
        out.append("")
        for coord in coords_as_strings:
            out.append(f"\\node[circle,draw,color=palletec{i % len(palette)},scale=0.5,line width=1.0] at {coord} {{}};")
    out.append("\end{tikzpicture}")
    with open("tikz_out.tikz", "w", encoding="utf-8") as f:
        f.writelines(out)


if __name__ == "__main__":
    CLI(main)