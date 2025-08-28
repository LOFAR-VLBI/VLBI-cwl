class: CommandLineTool
cwlVersion: v1.2
id: dp3_avg_intermediate
label: DP3 averaging for intermediate resolution imaging.
doc: |
  Average MeasurementSet in time, frequency and apply baseline dependent averaging for
  direction-dependent imaging at intermediate resolution in DDE-mode.

baseCommand: DP3

inputs:
  - id: msin
    type: Directory[]
    doc: Input MeasurementSet subbands.
    inputBinding:
      position: 0
      prefix: msin=
      separate: false
      itemSeparator: ','
      valueFrom: "[$(self.map(function(d) { return d.path || d.location; }).join(','))]"

outputs:
  - id: ms_avg
    doc: |
        The output data with corrected
        data in MeasurementSet format.
    type: Directory
    outputBinding:
      glob: concat_1asec.ms

  - id: logfile
    type: File[]
    outputBinding:
      glob: dp3_intermediate_avg*.log
    doc: |
        The files containing the stdout
        and stderr from the step.

arguments:
  - steps=[avg,bdaaverager]
  - avg.type=averager
  - avg.timeresolution=4
  - avg.freqresolution=48.82kHz
  - msout=concat_1asec.ms
  - bdaaverager.timebase=70e3
  - bdaaverager.maxinterval=32

requirements:
  - class: ShellCommandRequirement
  - class: InlineJavascriptRequirement

hints:
  - class: DockerRequirement
    dockerPull: vlbi-cwl
  - class: InitialWorkDirRequirement
    listing:
      - entry: $(inputs.msin)
  - class: ResourceRequirement
    coresMin: 12

stdout: dp3_intermediate_avg.log
stderr: dp3_intermediate_avg_err.log
