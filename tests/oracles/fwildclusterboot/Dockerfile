FROM rocker/r-ver:4.4.3@sha256:3dae5d2eeddf74f10e0a81fb6b7ae350295e288000304f438b844b2c1e00fe2c

WORKDIR /oracle

ARG RENV_VERSION=1.2.4
ARG RENV_SHA256=e63c637dc785d55848d9dbc6c9599378103803efd47c1f3f1f82057c00575e8c

RUN RENV_ARCHIVE="/tmp/renv_${RENV_VERSION}.tar.gz" \
    && R -e "download.file('https://cloud.r-project.org/src/contrib/renv_${RENV_VERSION}.tar.gz', '${RENV_ARCHIVE}', mode = 'wb', quiet = TRUE)" \
    && echo "${RENV_SHA256}  ${RENV_ARCHIVE}" | sha256sum --check --strict \
    && R CMD INSTALL "${RENV_ARCHIVE}" \
    && rm -f "${RENV_ARCHIVE}"

COPY renv.lock /oracle/renv.lock
RUN R -e 'renv::restore(lockfile = "/oracle/renv.lock", library = .libPaths()[1], prompt = FALSE)'
RUN R -e 'description <- utils::packageDescription("fwildclusterboot"); stopifnot(as.character(utils::packageVersion("fwildclusterboot")) == "0.14.3", description$RemoteSha == "336bb574eba169ac0183317f01d0564791d8122f")'

COPY run_reference.R /oracle/run_reference.R
COPY covariance_reference.R /oracle/covariance_reference.R

ENTRYPOINT ["Rscript", "/oracle/run_reference.R"]
