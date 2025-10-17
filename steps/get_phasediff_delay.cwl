class: CommandLineTool
cwlVersion: v1.2
id: get_phasediff
label: Polarization Phase Difference
doc: This step makes scalarphasediff solution files, needed for collecting source selection scores

baseCommand: facetselfcal

inputs:
    - id: phasediff_ms
      type: Directory
      doc: Input MeasurementSet
      inputBinding:
        position: 1
    - id: phaseup_config
      type: File
      doc: |
        Configuration file for facet_selfcal.
        Intended to be phaseup_config.txt.
      inputBinding:
        position: 0
        prefix: --configpath

outputs:
    - id: phasediff_h5out
      type: File
      doc: h5parm solution files with scalarphasediff solutions
      outputBinding:
        glob: "scalarphasediff*.h5"
    - id: scalarphase_h5out
      type: File
      doc: h5parm solution with simple scalarphase solutions
      outputBinding:
        glob: "scalarphase1*.h5"
    - id: phasediff_score
      type: File
      doc: csv with phasediff scores
      outputBinding:
        glob: phasediff_output.csv
    - id: logfile
      type: File[]
      doc: log files from facetselfcal
      outputBinding:
        glob: phasediff_delay*.log

requirements:
  - class: InlineJavascriptRequirement
  - class: InitialWorkDirRequirement
    listing:
      - entry: $(inputs.phasediff_ms)
        writable: true

hints:
  - class: DockerRequirement
    dockerPull: vlbi-cwl
  - class: ResourceRequirement
    coresMin: 2

stdout: phasediff_delay.log
stderr: phasediff_delay_err.log
