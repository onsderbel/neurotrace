'use strict';

const fs = require('fs');
const path = require('path');
const crypto = require('crypto');
const grpc = require('@grpc/grpc-js');
const { connect, signers } = require('@hyperledger/fabric-gateway');
const express = require('express');

const TN = process.env.FABRIC_TEST_NETWORK || path.join(process.env.HOME, 'fabric-samples/test-network');
const ORG1 = path.join(TN, 'organizations/peerOrganizations/org1.example.com');
const PEER_ENDPOINT = process.env.PEER_ENDPOINT || 'localhost:7051';
const PORT = Number(process.env.PORT || 3000);
const firstFile = (dir) => path.join(dir, fs.readdirSync(dir)[0]);
const deadline = (ms) => () => ({ deadline: Date.now() + ms });

// Message lisible : on préfère le message renvoyé par le chaincode
function errMsg(e) {
  if (Array.isArray(e.details) && e.details.length) return e.details.map((d) => d.message).join(' ; ');
  return e.message || String(e);
}
function httpStatus(msg) {
  if (/introuvable/i.test(msg)) return 404;
  if (/non autoris/i.test(msg)) return 403;
  if (/d[ée]j[àa] enregistr/i.test(msg)) return 409;
  if (/UNAVAILABLE|ECONNREFUSED|DEADLINE/i.test(msg)) return 503;
  return 400;
}

async function main() {
  const tlsCert = fs.readFileSync(path.join(ORG1, 'peers/peer0.org1.example.com/tls/ca.crt'));
  const userMsp = path.join(ORG1, 'users/User1@org1.example.com/msp');
  const credentials = fs.readFileSync(firstFile(path.join(userMsp, 'signcerts')));
  const privateKey = crypto.createPrivateKey(fs.readFileSync(firstFile(path.join(userMsp, 'keystore'))));

  const client = new grpc.Client(PEER_ENDPOINT, grpc.credentials.createSsl(tlsCert), {
    'grpc.ssl_target_name_override': 'peer0.org1.example.com',
  });
  const gateway = connect({
    client,
    identity: { mspId: 'Org1MSP', credentials },     // Org1 = organisation autorisée à écrire
    signer: signers.newPrivateKeySigner(privateKey),
    evaluateOptions: deadline(5000),
    endorseOptions: deadline(15000),
    submitOptions: deadline(5000),
    commitStatusOptions: deadline(60000),
  });
  const contract = gateway.getNetwork('mychannel').getContract('neurotrace');
  const decode = (bytes) => JSON.parse(Buffer.from(bytes).toString('utf8'));

  const app = express();
  app.use(express.json());

  // Accès réservé : si GATEWAY_API_KEY est définie, chaque requête (sauf /health) doit la fournir
  const API_KEY = process.env.GATEWAY_API_KEY;
  app.use((req, res, next) => {
    if (!API_KEY || req.path === '/health') return next();
    const given = Buffer.from(req.get('X-API-Key') || '');
    const wanted = Buffer.from(API_KEY);
    if (given.length !== wanted.length || !crypto.timingSafeEqual(given, wanted)) {
      return res.status(401).json({ error: 'Clé API manquante ou invalide' });
    }
    next();
  });

  const wrap = (fn) => async (req, res) => {
    try {
      const { status, body } = await fn(req);
      res.status(status).json(body);
    } catch (e) {
      const msg = errMsg(e);
      console.error(`[${req.method} ${req.path}] ${msg}`);
      res.status(httpStatus(msg)).json({ error: msg });
    }
  };

  app.get('/health', (req, res) => res.json({ status: 'ok' }));

  app.post('/artifacts', wrap(async (req) => ({
    status: 201,
    body: decode(await contract.submitTransaction('RegisterArtifact', JSON.stringify(req.body))),
  })));

  app.get('/artifacts/:id', wrap(async (req) => ({
    status: 200,
    body: decode(await contract.evaluateTransaction('GetArtifact', req.params.id)),
  })));

  app.get('/cases/:caseId/history', wrap(async (req) => ({
    status: 200,
    body: decode(await contract.evaluateTransaction('GetCaseHistory', req.params.caseId)),
  })));

  app.post('/artifacts/:id/verify', wrap(async (req) => ({
    status: 200,
    body: decode(await contract.evaluateTransaction('VerifyHash', req.params.id, String((req.body || {}).sha256 || ''))),
  })));

  // JSON mal formé dans le corps de la requête
  app.use((err, req, res, next) => res.status(400).json({ error: `Requête invalide : ${err.message}` }));

  app.listen(PORT, '127.0.0.1', () => console.log(`Gateway NeuroTrace sur http://127.0.0.1:${PORT}`));
}

main().catch((e) => { console.error('Démarrage impossible :', errMsg(e)); process.exit(1); });
