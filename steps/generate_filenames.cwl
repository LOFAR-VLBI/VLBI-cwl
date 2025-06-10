cwlVersion: v1.2
class: ExpressionTool
id: generate_filenames
label: Generate direction filenames
doc: |
    Take a MeasurementSet and a list of target sources, and
    creates a list of strings where the MeasurementSet name
    and target names are concatenated.

inputs:
    - id: msin
      type: Directory
      doc: A MeasurementSet to extract the name from.

    - id: source_ids
      type: string
      doc: A string containing a list of target source IDs.

outputs:
    - id: msout_names
      type: string
      doc: |
        a string containing the names for the MeasurementSets
        for each direction.

expression: |
  ${
    var msin = inputs.msin.basename.split(".")[0];
    var source_ids = inputs.source_ids.split(",");
    var list = [];
    for (var i = 0; i < source_ids.length; i++) {
      list.push(source_ids[i] + "_" + msin + ".mstargetphase");
    }

    return {"msout_names" : "[" + list.join(",") + "]"};
  }


requirements:
    - class: InlineJavascriptRequirement
