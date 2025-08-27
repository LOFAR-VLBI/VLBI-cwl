cwlVersion: v1.2
class: CommandLineTool
id: remove_flagged_stations
label: Remove fully flagged stations
doc: Removes from the MeasurementSet all stations for which all data are flagged.

baseCommand: remove_flagged_stations

inputs:
    - id: ms
      type: Directory
      doc: MeasurementSet
      inputBinding:
        position: 3

outputs:
    - id: cleaned_ms
      type: Directory
      doc: MeasurementSet where fully flagged stations are removed.
      outputBinding:
        glob: $( 'flagged_' + inputs.ms.basename )
    - id: logfile
      type: File[]
      doc: Log files from current step.
      outputBinding:
        glob: remove_flagged_stations*.log

requirements:
  - class: InlineJavascriptRequirement
  - class: InitialWorkDirRequirement
    listing:
      - entry: $(inputs.ms)
        writable: false

arguments:
  - --msout
  - $( 'flagged_' + inputs.ms.basename )

hints:
  - class: DockerRequirement
    dockerPull: vlbi-cwl
  - class: ResourceRequirement
    coresMin: 2

stdout: remove_flagged_stations.log
stderr: remove_flagged_stations_err.log
