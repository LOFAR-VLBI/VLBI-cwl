class: CommandLineTool
cwlVersion: v1.2
id: make-mslist
doc: |-
  Generates a text file containing the names of MeasurementSets, used by e.g. the DDF-pipeline.
  This will be the input for the subtract. This requires DDF-pipeline to be installed.

baseCommand: echo
stdout: big-mslist.txt

inputs:
  - id: ms
    type: Directory
    doc: Input MeasurementSet.
    inputBinding:
      position: 1
      valueFrom: $(self.basename)

outputs:
  - id: mslist
    type: stdout
    doc: Text file containing the names of the MeasurementSet.
