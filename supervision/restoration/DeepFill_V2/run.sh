cd ../../parsing/dml_csr/ || exit
python3 image_infer.py
cd ../../restoration/GPEN || exit
python3 drop_and_inpaint.py
cd ../DeepFill_V2 || exit
python3 test.py --image ../../parsing/dml_csr/infer_images/out/inpaint_st.jpg \
     --mask ../../parsing/dml_csr/infer_images/out/drop_st.png \
     --output ../../parsing/dml_csr/infer_images/out/output_st.png \
     --checkpoint_dir weights/release_celeba_hq_256_deepfill_v2
python3 test.py --image ../../parsing/dml_csr/infer_images/out/inpaint_ts.jpg \
     --mask ../../parsing/dml_csr/infer_images/out/drop_ts.png \
     --output ../../parsing/dml_csr/infer_images/out/output_ts.png \
     --checkpoint_dir weights/release_celeba_hq_256_deepfill_v2