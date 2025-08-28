cwlVersion: v1.2
class: ExpressionTool
requirements:
  - class: InlineJavascriptRequirement

label: Sort input entries by basename
doc: |
  Sorts an array of inputs according to their basename.
  Passed objects must thus have a basename attribute.

inputs:
  input_entry:
    type:
      - Any[]

outputs:
  sorted_entries:
    type:
      - Any[]

expression: |
  ${
    var sorted = inputs.input_entry.sort(function(a, b) {
      return a.basename.localeCompare(b.basename);
    });
    return {"sorted_entries": sorted};
  }
