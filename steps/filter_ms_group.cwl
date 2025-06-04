class: CommandLineTool
cwlVersion: v1.2
id: filter_ms_group
label: Filter MeasurementSet group
doc: |
    Collects MeasurementSets for concatenation,
    excluding dummy data.

baseCommand: filter_ms_group.py

arguments:
  # cwl.output.json is a privileged filename which, if it exists
  # in runtime.outdir, should be used by CWLas an output object.
  # The outputBinding fields of objects in outputs is ignored.
  # See https://www.commonwl.org/v1.2/CommandLineTool.html#Output_binding
  - valueFrom: $(runtime.outdir)/cwl.output.json
    prefix: --output_filename
    separate: true

inputs:
  - id: group_id
    type: string
    inputBinding:
      position: 1
      prefix: --group_id
      separate: true
    doc: |
        A string that determines which
        MeasurementSets should be combined.

  - id: groups_specification
    type: File
    inputBinding:
      position: 1
      prefix: --json_file
      separate: true
    doc: |
        A file containing directories of MeasurementSets.

  - id: measurement_sets
    type: Directory[]
    inputBinding:
      position: 2
    doc: The total number of input MeasurementSets.

outputs:
  - id: selected_ms
    type: string[]
    doc: The names of the selected MeasurementSets.

requirements:
  - class: InlineJavascriptRequirement
hints:
  - class: DockerRequirement
    dockerPull: vlbi-cwl

stdout: filter_ms_by_group.log
stderr: filter_ms_by_group_err.log
