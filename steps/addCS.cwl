cwlVersion: v1.2
class: CommandLineTool
id: addCS
doc: |
        Determines the station information from the data and
        adds back the core stations to the h5parm if the data has phased-up core stations.

baseCommand: h5_merger

inputs:
  - id: ms
    type: Directory
    doc: Input MeasurementSet data
    inputBinding:
      position: 2
      prefix: "-ms"
      separate: true
  - id: h5parm
    type: File
    doc: Input h5parm
    inputBinding:
      prefix: "-in"
      position: 3
      separate: true

outputs:
    - id: addCS_out_h5
      type: File
      doc: H5parm with preapplied solutions and core stations
      outputBinding:
        glob: $( inputs.h5parm.basename + '.addCS.h5' )
    - id: logfile
      type: File[]
      doc: Log files corresponding to this step
      outputBinding:
        glob: h5_merger_dd*.log

arguments:
  - --h5_out=$( inputs.h5parm.basename + '.addCS.h5' )
  - --add_ms_stations
  - --h5_time_freq=1

requirements:
  - class: InlineJavascriptRequirement

hints:
  - class: DockerRequirement
    dockerPull: vlbi-cwl

stdout: h5_merger_dd.log
stderr: h5_merger_dd_err.log
