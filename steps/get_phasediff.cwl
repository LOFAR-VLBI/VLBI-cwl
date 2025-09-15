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
        position: 20
        valueFrom: $(self.basename)

outputs:
    - id: phasediff_h5out
      type: File
      doc: h5parm solution files with scalarphasediff solutions
      outputBinding:
        glob: "scalarphasediff*.h5"
    - id: logfile
      type: File[]
      doc: log files from facetselfcal
      outputBinding:
        glob: phasediff*.log

requirements:
  - class: InlineJavascriptRequirement
  - class: InitialWorkDirRequirement
    listing:
      - entry: $(inputs.phasediff_ms)
        writable: true

arguments:
  - -i
  - phasediff
  - --forwidefield
  - --phaseupstations=core
  - --skipbackup
  - --uvmin=20000
  - --soltype-list=['scalarphasediff']
  - --solint-list=['10min']
  - --nchan-list=[6]
  - --docircular
  - --uvminscalarphasediff=0
  - --stop=1
  - --soltypecycles-list=[0]
  - --imsize=1600
  - --skymodelpointsource=1.0
  - --stopafterskysolve
  - --phasediff_only

hints:
  - class: DockerRequirement
    dockerPull: vlbi-cwl
  - class: ResourceRequirement
    coresMin: 2

stdout: phasediff.log
stderr: phasediff_err.log
