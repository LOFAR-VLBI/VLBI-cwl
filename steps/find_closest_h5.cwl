cwlVersion: v1.2
class: CommandLineTool
id: find_closest_h5
label: Find nearest direction from multi-dir h5parm
doc: Returns a h5parm which corresponds to the nearest direction from a multi-direction h5parm

baseCommand: find_closest_h5

inputs:
  - id: ms
    type: Directory
    doc: Input MeasurementSet
    inputBinding:
      prefix: "--msin"
      position: 3
      separate: true
  - id: h5parm
    type: File?
    doc: Input h5parm
    inputBinding:
      prefix: "--h5_in"
      position: 2
      separate: true

outputs:
    - id: closest_h5
      type: File
      doc: Output h5parm
      outputBinding:
        glob: output_h5s/source_0.h5
    - id: logfile
      type: File[]
      doc: Log files corresponding to this step
      outputBinding:
        glob: applycal_dd*.log


requirements:
  - class: InlineJavascriptRequirement

hints:
  - class: DockerRequirement
    dockerPull: vlbi-cwl

stdout: find_closest_h5.log
stderr: find_closest_h5_err.log