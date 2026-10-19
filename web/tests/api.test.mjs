import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import test from 'node:test';
import ts from 'typescript';

const source = readFileSync(new URL('../src/api.ts', import.meta.url), 'utf8')
  .replace('import.meta.env.VITE_API_BASE_URL', JSON.stringify('https://api.example.test'));
const code = ts.transpileModule(source, { compilerOptions: { module: ts.ModuleKind.ESNext, target: ts.ScriptTarget.ES2022 } }).outputText;
const { apiFetch, fetchSourceDocument } = await import('data:text/javascript;base64,' + Buffer.from(code).toString('base64'));
globalThis.localStorage = { getItem: () => 'test-session' };

test('protected downloads use bearer headers, never URL tokens', async () => {
  let request;
  globalThis.window = { fetch: async (url, options) => {
    request = { url, options };
    return new Response(new Blob(['source image'], { type: 'image/png' }));
  } };
  const document = await fetchSourceDocument('/invoices/catalog/download?invoice_id=1&token=old-session');
  assert.equal(request.options.headers.get('Authorization'), 'Bearer test-session');
  assert.equal(request.url.searchParams.has('token'), false);
  assert.equal(request.options.cache, 'no-store');
  assert.ok(document.url.startsWith('blob:'));
  assert.equal(await (await fetch(document.url)).text(), 'source image');
  document.release();
  await assert.rejects(fetch(document.url));
});

test('denied source requests cannot become document previews', async () => {
  globalThis.window = { fetch: async () => new Response('denied', { status: 401 }) };
  await assert.rejects(fetchSourceDocument('/invoices/catalog/download?invoice_id=1'), /session/);
});

test('bearer tokens are never sent to another origin', async () => {
  globalThis.window = { fetch: async (url, options) => {
    assert.equal(options.headers.has('Authorization'), false);
    return new Response('ok');
  } };
  await apiFetch('https://other.example.test/file');
});

test('document cancellation signal is forwarded', async () => {
  const controller = new AbortController();
  globalThis.window = { fetch: async (url, options) => {
    assert.equal(options.signal, controller.signal);
    return new Response(new Blob(['pdf'], { type: 'application/pdf' }));
  } };
  const document = await fetchSourceDocument('/invoices/catalog/download', controller.signal);
  document.release();
});
