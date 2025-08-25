cwlVersion: v1.2
class: CommandLineTool
id: applycal
doc: Applies calibration solutions from h5parm to MeasurementSet.

baseCommand:
  - python3

inputs:
    - id: ms
      type: Directory[]
      doc: Input MeasurementSet
      inputBinding:
        position: 5
    - id: h5parm
      type: File?
      doc: Input h5parm to be applied
      inputBinding:
        prefix: "--h5"
        position: 4
        separate: true
    - id: lofar_helpers
      type: Directory
      doc: lofar helpers directory

outputs:
    - id: ms_out
      type: Directory[]
      doc: Output MeasurementSet with solutions applied
      outputBinding:
        glob: "applied_*"
    - id: logfile
      type: File[]
      doc: Log files corresponding to this step
      outputBinding:
        glob: applycal*.log

requirements:
  - class: InlineJavascriptRequirement
  - class: InitialWorkDirRequirement
    listing:
      - entry: $(inputs.ms)
      - entry: $(inputs.h5parm)

arguments:
  - $( inputs.lofar_helpers.path + '/ms_helpers/applycal.py' )
  - --msout=$( 'applied_' + inputs.ms.basename )

hints:
  - class: DockerRequirement
    dockerPull: vlbi-cwl
  - class: ResourceRequirement
    coresMin: 4

stdout: applycal.log
stderr: applycal_err.log