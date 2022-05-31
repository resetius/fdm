project(superlu)

set(DIR ${CMAKE_CURRENT_SOURCE_DIR}/../superlu/SRC)

set(ALLAUX
    ${DIR}/superlu_timer.c
    ${DIR}/util.c
    ${DIR}/memory.c
    ${DIR}/get_perm_c.c
    ${DIR}/mmd.c
    ${DIR}/sp_coletree.c
    ${DIR}/sp_preorder.c
    ${DIR}/sp_ienv.c
    ${DIR}/relax_snode.c
    ${DIR}/heap_relax_snode.c
    ${DIR}/colamd.c
    ${DIR}/ilu_relax_snode.c
    ${DIR}/ilu_heap_relax_snode.c
    ${DIR}/mark_relax.c
    ${DIR}/mc64ad.c
    ${DIR}/qselect.c
    ${DIR}/input_error.c
    ${DIR}/dmach.c
    ${DIR}/smach.c
    )

set(SLUSRC
	${DIR}/sgssv.c
    ${DIR}/sgssvx.c
	${DIR}/ssp_blas2.c
    ${DIR}/ssp_blas3.c
    ${DIR}/sgscon.c
	${DIR}/slangs.c
    ${DIR}/sgsequ.c
    ${DIR}/slaqgs.c
    ${DIR}/spivotgrowth.c
	${DIR}/sgsrfs.c
    ${DIR}/sgstrf.c
    ${DIR}/sgstrs.c
    ${DIR}/scopy_to_ucol.c
	${DIR}/ssnode_dfs.c
    ${DIR}/ssnode_bmod.c
	${DIR}/spanel_dfs.c
    ${DIR}/spanel_bmod.c
	${DIR}/sreadhb.c
    ${DIR}/sreadrb.c
    ${DIR}/sreadtriple.c
    ${DIR}/sreadMM.c
	${DIR}/scolumn_dfs.c
    ${DIR}/scolumn_bmod.c
    ${DIR}/spivotL.c
    ${DIR}/spruneL.c
	${DIR}/smemory.c
    ${DIR}/sutil.c
    ${DIR}/smyblas2.c
    ${DIR}/sgsisx.c
    ${DIR}/sgsitrf.c
    ${DIR}/sldperm.c
	${DIR}/ilu_sdrop_row.c
    ${DIR}/ilu_ssnode_dfs.c
	${DIR}/ilu_scolumn_dfs.c
    ${DIR}/ilu_spanel_dfs.c
    ${DIR}/ilu_scopy_to_ucol.c
	${DIR}/ilu_spivotL.c
    ${DIR}/sdiagonal.c
    ${DIR}/slacon2.c
)

set(DLUSRC
    ${DIR}/dgssv.c
    ${DIR}/dgssvx.c
    ${DIR}/dsp_blas2.c
    ${DIR}/dsp_blas3.c
    ${DIR}/dgscon.c
    ${DIR}/dlangs.c
    ${DIR}/dgsequ.c
    ${DIR}/dlaqgs.c
    ${DIR}/dpivotgrowth.c
    ${DIR}/dgsrfs.c
    ${DIR}/dgstrf.c
    ${DIR}/dgstrs.c
    ${DIR}/dcopy_to_ucol.c
    ${DIR}/dsnode_dfs.c
    ${DIR}/dsnode_bmod.c
    ${DIR}/dpanel_dfs.c
    ${DIR}/dpanel_bmod.c
    ${DIR}/dreadhb.c
    ${DIR}/dreadrb.c
    ${DIR}/dreadtriple.c
    ${DIR}/dreadMM.c
    ${DIR}/dcolumn_dfs.c
    ${DIR}/dcolumn_bmod.c
    ${DIR}/dpivotL.c
    ${DIR}/dpruneL.c
    ${DIR}/dmemory.c
    ${DIR}/dutil.c
    ${DIR}/dmyblas2.c
    ${DIR}/dgsisx.c
    ${DIR}/dgsitrf.c
    ${DIR}/dldperm.c
    ${DIR}/ilu_ddrop_row.c
    ${DIR}/ilu_dsnode_dfs.c
    ${DIR}/ilu_dcolumn_dfs.c
    ${DIR}/ilu_dpanel_dfs.c
    ${DIR}/ilu_dcopy_to_ucol.c
    ${DIR}/ilu_dpivotL.c
    ${DIR}/ddiagonal.c
    ${DIR}/dlacon2.c
    ${DIR}/dGetDiagU.c)


add_library(superlu SHARED ${ALLAUX} ${SLUSRC} ${DLUSRC})

target_include_directories(superlu PUBLIC ${DIR})
target_compile_options(superlu PRIVATE -DVENDOR_BLAS)

target_link_directories(superlu PUBLIC ${BLAS_LIBRARY_DIRS})
target_link_libraries(superlu PUBLIC ${BLAS_LIBRARIES})

