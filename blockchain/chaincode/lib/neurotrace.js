'use strict';

const { Contract } = require('fabric-contract-api');

// L'ordre de ce tableau est l'ordre de la chaîne de provenance
const ARTIFACT_TYPES = [
  'ORIGINAL_IMAGE', 'PREPROCESSED_IMAGE', 'FEATURES',
  'PREDICTION', 'VALIDATION', 'EXPLANATION', 'FINAL_RESULT',
];
const WRITER_MSP = 'Org1MSP';          // seule organisation autorisée à enregistrer
const ART_PREFIX = 'artifact';         // clé composite : artifact + [artifact_id]
const CASE_INDEX = 'case~artifact';    // index : [case_id, artifact_id]
const STATUS_OK = 'INTEGRITY_VERIFIED';
const STATUS_KO = 'MODIFICATION_DETECTED';

class NeuroTrace extends Contract {
  constructor() {
    super('NeuroTrace');
  }

  // Écriture : enregistre un artefact (jamais modifié ni écrasé ensuite)
  async RegisterArtifact(ctx, artifactJson) {
    const msp = ctx.clientIdentity.getMSPID();
    if (msp !== WRITER_MSP) {
      throw new Error(`Organisation ${msp} non autorisée à enregistrer un artefact`);
    }

    let input;
    try {
      input = JSON.parse(artifactJson);
    } catch (e) {
      throw new Error(`JSON invalide : ${e.message}`);
    }
    this._validate(input);

    const key = ctx.stub.createCompositeKey(ART_PREFIX, [input.artifact_id]);
    const existing = await ctx.stub.getState(key);
    if (existing && existing.length > 0) {
      throw new Error(`Artefact ${input.artifact_id} déjà enregistré : écrasement interdit`);
    }

    const parentId = input.parent_artifact_id || null;
    if (input.artifact_type === 'ORIGINAL_IMAGE') {
      if (parentId) throw new Error('ORIGINAL_IMAGE ne peut pas avoir de parent');
    } else {
      if (!parentId) throw new Error(`parent_artifact_id obligatoire pour ${input.artifact_type}`);
      const parent = await this._read(ctx, parentId);
      if (parent.case_id !== input.case_id) {
        throw new Error(`Le parent ${parentId} appartient à un autre cas`);
      }
    }

    const artifact = {
      artifact_id: input.artifact_id,
      case_id: input.case_id,
      artifact_type: input.artifact_type,
      sha256: input.sha256.toLowerCase(),
      timestamp: this._txTime(ctx),          // heure de la transaction (fixée par le client, identique sur tous les peers)
      model_version: input.model_version || null,
      parent_artifact_id: parentId,
      storage_ref: input.storage_ref,
    };

    await ctx.stub.putState(key, Buffer.from(JSON.stringify(artifact)));
    const idx = ctx.stub.createCompositeKey(CASE_INDEX, [artifact.case_id, artifact.artifact_id]);
    await ctx.stub.putState(idx, Buffer.from('\u0000'));

    return JSON.stringify(artifact);
  }

  // Lecture d'un artefact
  async GetArtifact(ctx, artifactId) {
    return JSON.stringify(await this._read(ctx, artifactId));
  }

  // Historique d'un cas, dans l'ordre de la chaîne de provenance
  async GetCaseHistory(ctx, caseId) {
    const iterator = await ctx.stub.getStateByPartialCompositeKey(CASE_INDEX, [caseId]);
    const artifacts = [];
    let res = await iterator.next();
    while (!res.done) {
      const { attributes } = ctx.stub.splitCompositeKey(res.value.key);
      artifacts.push(await this._read(ctx, attributes[1]));
      res = await iterator.next();
    }
    await iterator.close();
    const rank = (a) => ARTIFACT_TYPES.indexOf(a.artifact_type);   // ordre de la chaîne de provenance
    artifacts.sort((a, b) => rank(a) - rank(b) || a.timestamp.localeCompare(b.timestamp));
    return JSON.stringify(artifacts);
  }

  // Vérification : compare le hash fourni au hash enregistré
  async VerifyHash(ctx, artifactId, sha256) {
    if (!/^[0-9a-fA-F]{64}$/.test(sha256 || '')) {
      throw new Error('Hash invalide : 64 caractères hexadécimaux attendus');
    }
    const artifact = await this._read(ctx, artifactId);
    const provided = sha256.toLowerCase();
    return JSON.stringify({
      artifact_id: artifactId,
      status: artifact.sha256 === provided ? STATUS_OK : STATUS_KO,
      expected_sha256: artifact.sha256,
      provided_sha256: provided,
    });
  }

  // ----- fonctions internes -----

  async _read(ctx, artifactId) {
    const key = ctx.stub.createCompositeKey(ART_PREFIX, [artifactId]);
    const data = await ctx.stub.getState(key);
    if (!data || data.length === 0) throw new Error(`Artefact ${artifactId} introuvable`);
    return JSON.parse(data.toString());
  }

  _validate(a) {
    if (!a || typeof a !== 'object') throw new Error('Artefact manquant');
    if (!/^[A-Za-z0-9_-]{1,100}$/.test(a.artifact_id || '')) throw new Error('artifact_id invalide');
    if (!/^CASE_\d{3,}$/.test(a.case_id || '')) throw new Error('case_id invalide (format CASE_001)');
    if (!ARTIFACT_TYPES.includes(a.artifact_type)) {
      throw new Error(`artifact_type invalide (attendu : ${ARTIFACT_TYPES.join(', ')})`);
    }
    if (!/^[0-9a-fA-F]{64}$/.test(a.sha256 || '')) throw new Error('sha256 invalide (64 caractères hexadécimaux)');
    if (!a.storage_ref || typeof a.storage_ref !== 'string') throw new Error('storage_ref obligatoire');
  }

  _txTime(ctx) {
    const ts = ctx.stub.getTxTimestamp();
    const ms = Number(ts.seconds.toString()) * 1000 + Math.floor(ts.nanos / 1e6);
    return new Date(ms).toISOString();
  }
}

module.exports = NeuroTrace;
