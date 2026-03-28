"""
Final clean evaluation: dense KB, ensemble fusion, all datasets.
No calibration overhead -- just run and report.
"""

import os, time, warnings
import numpy as np
import pandas as pd
from datasets_loader import load_dataset, DATASET_CONFIGS
from embeddings import ChronosEmbeddingExtractor
from knowledge_base import SelfRetrievalKB, KBEntry
from fusion import ForecastEnsembleFusion, ShapeBlendFusion
import faiss

warnings.filterwarnings('ignore')


def build_kb(extractor, data, ws, pl, stride=48):
    entries, preds = [], {}
    for start in range(0, len(data) - ws - pl, stride):
        w, f = data[start:start+ws], data[start+ws:start+ws+pl]
        if len(f) < pl: continue
        emb = extractor.extract_embedding(w)
        pred = extractor.predict_median(w, pl)
        entry = KBEntry(window=w, future=f, embedding=emb, timestamp=start,
                       uncertainty=0, error=float(np.mean((pred-f)**2)))
        entries.append(entry)
        preds[id(entry)] = pred
    return entries, preds


def eval_test(extractor, test, ws, pl, kb_entries, fusion=None):
    kb = SelfRetrievalKB(extractor.embedding_dim)
    for e in kb_entries: kb.add(e)
    mses, maes = [], []
    for start in range(0, len(test)-ws-pl+1, pl):
        ctx, gt = test[start:start+ws], test[start+ws:start+ws+pl]
        med, _, unc = extractor.predict_with_uncertainty(ctx, pl)
        fc = med
        if fusion is not None and kb.size > 0:
            emb = extractor.extract_embedding(ctx)
            ret = kb.retrieve(emb, k=5, use_error_weighting=False)
            if ret:
                fc = fusion.fuse(med, unc, ret, pl, current_context=ctx)
        mses.append(float(np.mean((fc-gt)**2)))
        maes.append(float(np.mean(np.abs(fc-gt))))
    return float(np.mean(mses)), float(np.mean(maes)), len(mses)


def run_all(data_dir='./data', results_dir='./results',
            model_name='amazon/chronos-bolt-small', device='cuda', datasets=None):
    os.makedirs(results_dir, exist_ok=True)
    extractor = ChronosEmbeddingExtractor(model_name, device)
    print(f"Model loaded. dim={extractor.embedding_dim}")

    if datasets is None:
        datasets = ['ETTh1', 'ETTh2', 'ETTm1', 'ETTm2', 'Weather', 'ECL', 'Traffic']

    all_results = []
    for ds in datasets:
        train, val, test, config = load_dataset(ds, data_dir)
        tv = np.concatenate([train, val])

        for pl in config['prediction_lengths']:
            print(f"\n{ds} H={pl}")
            t0 = time.time()
            kb, kbp = build_kb(extractor, tv, 512, pl, stride=256)
            print(f"  KB: {len(kb)} entries ({time.time()-t0:.0f}s)")

            # Baseline
            t0 = time.time()
            bl_mse, bl_mae, nw = eval_test(extractor, test, 512, pl, kb, fusion=None)
            print(f"  baseline:      MSE={bl_mse:.4f}  MAE={bl_mae:.4f}  ({time.time()-t0:.0f}s)")

            all_results.append({'dataset':ds,'pred_len':pl,'method':'baseline',
                               'mse':bl_mse,'mae':bl_mae,'n_windows':nw})

            # Ensemble with different base weights
            for bw in [2.0, 3.0, 5.0]:
                fusion_e = ForecastEnsembleFusion(extractor, base_weight=bw)
                t0 = time.time()
                e_mse, e_mae, _ = eval_test(extractor, test, 512, pl, kb, fusion=fusion_e)
                imp = (bl_mse-e_mse)/bl_mse*100
                print(f"  ensemble_bw{int(bw)}:  MSE={e_mse:.4f}  ({imp:+.2f}%)  ({time.time()-t0:.0f}s)")
                all_results.append({'dataset':ds,'pred_len':pl,'method':f'ensemble_bw{int(bw)}',
                                   'mse':e_mse,'mae':e_mae,'n_windows':nw})

            # Shape blend
            fusion_s = ShapeBlendFusion(extractor, alpha=0.5)
            sb_mse, sb_mae, _ = eval_test(extractor, test, 512, pl, kb, fusion=fusion_s)
            imp_s = (bl_mse-sb_mse)/bl_mse*100
            print(f"  shape_blend:   MSE={sb_mse:.4f}  ({imp_s:+.2f}%)")

            all_results.append({'dataset':ds,'pred_len':pl,'method':'shape_blend',
                               'mse':sb_mse,'mae':sb_mae,'n_windows':nw})

            # Save incrementally
            pd.DataFrame(all_results).to_csv(
                os.path.join(results_dir, 'ablation_final.csv'), index=False)

    # Streaming
    for ds in [d for d in ['ETTh1','ETTm2','Weather'] if d in datasets]:
        print(f"\nStreaming: {ds}")
        train, val, test, _ = load_dataset(ds, data_dir)
        full = np.concatenate([train, val, test])
        kb = SelfRetrievalKB(extractor.embedding_dim)
        fusion = ForecastEnsembleFusion(extractor, base_weight=2.0)
        cmse, wmse, blmse = [], [], []
        positions = list(range(512, len(full)-96, 96))
        for step, t in enumerate(positions):
            ctx, gt = full[t-512:t], full[t:t+96]
            med, _, unc = extractor.predict_with_uncertainty(ctx, 96)
            fc = med.copy()
            blmse.append(float(np.mean((med-gt)**2)))
            if kb.size > 0:
                emb = extractor.extract_embedding(ctx)
                ret = kb.retrieve(emb, k=5, use_error_weighting=False)
                if ret: fc = fusion.fuse(med, unc, ret, 96, current_context=ctx)
            wmse.append(float(np.mean((fc-gt)**2)))
            cmse.append(float(np.mean(wmse)))
            emb = extractor.extract_embedding(ctx)
            entry = KBEntry(window=ctx.copy(),future=gt.copy(),embedding=emb,
                           timestamp=step,uncertainty=unc,
                           error=float(np.mean((med-gt)**2)))
            kb.add(entry)
            if (step+1)%50==0 or (step+1)==len(positions):
                print(f"  {step+1}/{len(positions)}: ASR={cmse[-1]:.4f} bl={np.mean(blmse):.4f}")
        np.save(os.path.join(results_dir,f'streaming_final_{ds}.npy'),
               {'cumulative_mse':cmse,'window_mses':wmse,'baseline_mses':blmse})
        imp = (np.mean(blmse)-cmse[-1])/np.mean(blmse)*100
        print(f"  {ds}: ASR={cmse[-1]:.4f} bl={np.mean(blmse):.4f} ({imp:+.2f}%)")


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--data_dir', default='./data')
    p.add_argument('--results_dir', default='./results')
    p.add_argument('--model', default='amazon/chronos-bolt-small')
    p.add_argument('--device', default='cuda')
    p.add_argument('--datasets', nargs='+', default=None)
    a = p.parse_args()
    run_all(a.data_dir, a.results_dir, a.model, a.device, a.datasets)
