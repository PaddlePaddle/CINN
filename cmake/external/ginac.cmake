find_program(AUTORECONF autoreconf)
if(NOT AUTORECONF)
	message(SEND_ERROR "Can not build GiNaC, missing binary for autoreconf")
endif()

find_program(PYTHON2 python2)
if(NOT PYTHON2)
	message(SEND_ERROR "Can not build GiNaC, missing binary for python2")
endif()

ExternalProject_Add(
    GiNaC-EP
    GIT_REPOSITORY "git://www.ginac.de/ginac.git"
	GIT_TAG "release_1-7-8"
	DOWNLOAD_NO_PROGRESS 1
	UPDATE_COMMAND ""
	CONFIGURE_COMMAND ${AUTORECONF} -iv <SOURCE_DIR> 
		COMMAND <SOURCE_DIR>/configure --quiet --prefix=<INSTALL_DIR> PYTHON=${PYTHON2} PKG_CONFIG_PATH=<INSTALL_DIR>/lib/pkgconfig/
	BUILD_COMMAND ${CMAKE_MAKE_PROGRAM} -C ginac
	INSTALL_COMMAND ${CMAKE_MAKE_PROGRAM} -C ginac install
	LOG_INSTALL 1
)

ExternalProject_Get_Property(GiNaC-EP INSTALL_DIR)

add_library(GINAC SHARED "${INSTALL_DIR}/lib/libginac${DYNAMIC_EXT}" "${INSTALL_DIR}/include")
add_library(GINAC STATIC "${INSTALL_DIR}/lib/libginac${STATIC_EXT}" "${INSTALL_DIR}/include")

add_dependencies(GiNaC-EP CLN_SHARED CLN_STATIC)
add_dependencies(GINAC_SHARED GiNaC-EP)
add_dependencies(GINAC_STATIC GiNaC-EP)
add_dependencies(resources GINAC_SHARED GINAC_STATIC)

mark_as_advanced(AUTORECONF)
