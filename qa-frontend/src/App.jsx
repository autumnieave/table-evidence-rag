import React, { useState, useRef, useEffect } from 'react';
import './App.css';
import ChartView from './ChartView';
import Markdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import rehypeHighlight from 'rehype-highlight';
import 'highlight.js/styles/github.css'; // 代码高亮样式

// VITE_API_URL 为空串时走同源相对路径（docker 版经 Nginx 反代到后端）；本地 dev 由 qa-frontend/.env 提供 http://localhost:8000
const API_URL = import.meta.env.VITE_API_URL ?? '';
const STORAGE_KEY = 'chatHistory';
const SESSIONS_KEY = 'chatSessionsV2';
const MAX_SESSIONS = 5;

// 演示模式：答案正文与引用原文按数据版权脱敏，仅保留界面结构（VITE_DEMO_MASK=1 或 URL 带 ?demo=1）
const DEMO_MASK = import.meta.env.VITE_DEMO_MASK === '1'
  || new URLSearchParams(window.location.search).has('demo');
const MASK_REF_TEXT = '引用内容';
const MASK_REF_NAME = '引用来源';
const MASK_REF_IMAGE = '引用配图';
const mask = (value, placeholder) => (DEMO_MASK ? placeholder : value);

// 研报原文为 LaTeX 转换产物，含 $1 8 . 9 0 \%$ 这类排版残留；
// 展示前轻量清洗，避免用户读到乱码级上下文。只作用于「数字命中位置」的上下文展示，
// 不改动后端核验口径，也不动「原文片段」正文。
const deLatex = (value) => (value || '')
  .replace(/\$([^$]{1,120})\$/g, (_m, body) => body.replace(/\s+/g, ''))
  .replace(/\$([^$]*)$/, (_m, body) => body.replace(/\s+/g, ''))
  .replace(/\\([%$&#_{}])/g, '$1')
  .replace(/\\[a-zA-Z]+/g, ' ')
  .replace(/[{}]/g, '')
  .replace(/\\/g, '')
  .replace(/\s+/g, ' ')
  .trim();

// 演示模式下的正文呈现：固定五行高，把「演示内容」放在中间那行的正中间（水平+垂直居中）。
// 目的只是把版式撑起来；配合右上角「演示模式」角标即可读懂。
const DemoAnswer = () => (
  <div
    className="grid h-[120px] rounded-xl border border-amber-200 bg-amber-50/40"
    style={{ gridTemplateRows: 'repeat(5, minmax(0, 1fr))' }}
  >
    <div className="row-start-3 flex items-center justify-center text-sm text-amber-700 select-none">
      演示内容
    </div>
  </div>
);

// 会话工具：每个会话独立 id / user_id（隔离后端记忆），最多 MAX_SESSIONS 个
const genId = () => 's-' + Date.now() + '-' + Math.random().toString(36).slice(2, 8);
const makeSession = (messages = []) => {
  const id = genId();
  const firstUser = (messages || []).find(m => m.role === 'user');
  return {
    id,
    title: (firstUser?.content || '新对话').slice(0, 18),
    userId: 'session-' + id,
    messages: messages || [],
  };
};

// 图片路径归一化：后端返回 /result/xxx（相对）或 http(s) 直链
const resolveImageUrl = (url) => {
  if (!url) return '';
  return url.startsWith('http') ? url : `${API_URL}${url}`;
};

// 从 Windows/Linux 路径中取文件名
const basename = (p) => {
  if (!p) return '';
  const parts = String(p).split(/[\\/]+/).filter(Boolean);
  return parts[parts.length - 1] || p;
};

const formatTime = (ts) => {
  if (!ts) return '';
  try {
    return new Date(ts).toLocaleTimeString('zh-CN', { hour: '2-digit', minute: '2-digit' });
  } catch {
    return '';
  }
};

// 模式定义
const MODES = [
  { key: 'agent', label: 'Agent 分析', desc: '查数字和对比 · 查财务数据库并生成图表，也支持研报', placeholder: '查财务数字、做对比分析，如「对比东阿阿胶与云南白药的盈利能力」', active: 'bg-green-500', badge: 'bg-green-100 text-green-600' },
];

// 后端 Agent 阶段事件 → 前端提示文案
const STAGE_COPY = {
  parse: '正在分析问题…',
  thinking: '正在思考下一步…',
  search_reports: '正在检索研报内容…',
  query_financial: '正在查询财务数据并生成图表（约 1-2 分钟）…',
  generate: '正在生成答案…',
};

// 各模式推荐问题（点击即发送）
const SUGGESTIONS = {
  agent: [
    { icon: '📊', q: '分析万邦德2023年营收结构并生成图表' },
    { icon: '⚖️', q: '对比东阿阿胶与云南白药的盈利能力' },
    { icon: '📄', q: '药明康德2024年研报主要观点是什么？' },
    { icon: '📈', q: '云南白药近三年净利润趋势如何？' },
  ],
};

// 引用来源核验标记：只暴露用户能理解的一件事——来源文件是否已核验存在。
// exact（路径精确命中）与 fuzzy（按文件名/目录尾段匹配）都属已核验，不再挂牌自证；
// 也不暴露 exact/fuzzy 这类实现术语，避免每条引用都挂告警色造成误导。
const CitationBadge = ({ citation }) => {
  if (!citation) return null;
  const map = {
    fuzzy: ['bg-gray-100 text-gray-500', '来源已核对', '文件名与语料库中的研报一致'],
    missing: ['bg-red-100 text-red-700', '来源未核验', '引用文件在语料库中不存在'],
  };
  const entry = map[citation.status];
  if (!entry) return null;
  const [cls, label, tip] = entry;
  return (
    <span className={`inline-block px-1.5 py-0.5 rounded text-[11px] font-medium whitespace-nowrap ${cls}`} title={tip}>
      {label}
    </span>
  );
};

// 从 localStorage 读取会话列表（优先新格式，兼容旧版单会话历史）
const loadSessions = () => {
  try {
    const stored = localStorage.getItem(SESSIONS_KEY);
    if (stored) {
      const parsed = JSON.parse(stored);
      if (Array.isArray(parsed) && parsed.length > 0) {
        return parsed
          .filter(s => s && Array.isArray(s.messages))
          .map(s => ({ ...s, messages: (s.messages || []).filter(m => !m.isStreaming) }))
          .slice(-MAX_SESSIONS);
      }
    }
    const legacy = localStorage.getItem(STORAGE_KEY);
    if (legacy) {
      const parsed = JSON.parse(legacy);
      if (Array.isArray(parsed) && parsed.length > 0) {
        return [makeSession(parsed.filter(m => !m.isStreaming))];
      }
    }
  } catch (e) {
    console.warn('读取会话失败:', e);
  }
  return [makeSession()];
};

// 保存会话列表到 localStorage（每会话最多 50 条）
const saveSessions = (sessions) => {
  try {
    const toStore = sessions.map(s => ({
      ...s,
      messages: (s.messages || []).filter(m => !m.isStreaming).slice(-50),
    }));
    localStorage.setItem(SESSIONS_KEY, JSON.stringify(toStore));
  } catch (e) {
    console.warn('保存会话失败:', e);
  }
};


function App() {
  const [sessions, setSessions] = useState(loadSessions);
  const [activeId, setActiveId] = useState(null);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [mode] = useState('agent'); // 前端固定 Agent 模式；自研 RAG / 多轮澄清仅供 CLI 本地测试
  const [stage, setStage] = useState(null); // 后端阶段事件（agent 等待期反馈）
  const [health, setHealth] = useState({ status: 'checking', collection: '' });
  const [copiedId, setCopiedId] = useState(null);
  const [previewImage, setPreviewImage] = useState(null);
  const [confirmClear, setConfirmClear] = useState(false);
  const [sessionHint, setSessionHint] = useState(false);
  const messagesEndRef = useRef(null);
  const taRef = useRef(null);
  const abortRef = useRef(null);

  const activeSession = sessions.find(s => s.id === activeId) || sessions[0] || null;
  const messages = activeSession?.messages || [];

  // 会话级更新工具：消息只存在于当前会话（列表为唯一数据源）
  const patchActiveMessages = (fn) => {
    setSessions(prev => {
      const target = prev.find(s => s.id === activeId) || prev[0];
      if (!target) return prev;
      return prev.map(s => (s.id === target.id ? { ...s, messages: fn(s.messages || []) } : s));
    });
  };
  const appendMessages = (...newMsgs) => patchActiveMessages(msgs => [...msgs, ...newMsgs]);
  const updateMessage = (id, patch) => patchActiveMessages(msgs => msgs.map(m => (m.id === id ? { ...m, ...patch } : m)));

  // 后端健康检查（每 15 秒轮询）
  useEffect(() => {
    let cancelled = false;
    const check = async () => {
      try {
        const res = await fetch(`${API_URL}/health`);
        const data = await res.json();
        if (!cancelled) setHealth({ status: 'ok', collection: data.collection || '' });
      } catch (e) {
        if (!cancelled) setHealth({ status: 'down', collection: '' });
      }
    };
    check();
    const timer = setInterval(check, 15000);
    return () => { cancelled = true; clearInterval(timer); };
  }, []);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  // 自动持久化会话列表到 localStorage
  useEffect(() => {
    saveSessions(sessions);
  }, [sessions]);

  // 输入框自动增高（最高 160px）
  useEffect(() => {
    const ta = taRef.current;
    if (!ta) return;
    ta.style.height = 'auto';
    ta.style.height = Math.min(ta.scrollHeight, 160) + 'px';
  }, [input]);

  const copyText = async (id, text) => {
    try {
      let ok = false;
      if (navigator.clipboard && window.isSecureContext) {
        try {
          await navigator.clipboard.writeText(text);
          ok = true;
        } catch (e) {
          ok = false;
        }
      }
      if (!ok) {
        const ta = document.createElement('textarea');
        ta.value = text;
        ta.style.position = 'fixed';
        ta.style.left = '-9999px';
        ta.style.top = '0';
        document.body.appendChild(ta);
        ta.focus();
        ta.select();
        ok = document.execCommand('copy');
        document.body.removeChild(ta);
      }
      if (ok) {
        setCopiedId(id);
        setTimeout(() => setCopiedId(null), 1500);
      } else {
        console.warn('复制失败: 浏览器未授权剪贴板');
      }
    } catch (e) {
      console.warn('复制失败:', e);
    }
  };

  const newSession = () => {
    if (isLoading) return;
    if (sessions.length >= MAX_SESSIONS) {
      setSessionHint(true);
      setTimeout(() => setSessionHint(false), 2000);
      return;
    }
    const s = makeSession();
    setSessions(prev => [...prev, s]);
    setActiveId(s.id);
    setInput('');
    setStage(null);
  };

  const deleteSession = (id) => {
    if (isLoading) return;
    setSessions(prev => {
      const next = prev.filter(s => s.id !== id);
      return next.length ? next : [makeSession()];
    });
    if (id === activeId) setActiveId(null);
    setStage(null);
  };

  const stopGeneration = () => {
    abortRef.current?.abort();
  };

  const sendMessage = async (textOverride) => {
    const question = (textOverride ?? input).trim();
    if (!question || isLoading) return;

    const userMsg = { id: `u-${Date.now()}`, role: 'user', content: question, ts: Date.now() };
    const aiMsgId = `a-${Date.now()}`;
    const aiMsg = {
      role: 'assistant',
      content: '',
      image: [],
      references: [],
      chart_json: null,
      id: aiMsgId,
      ts: Date.now(),
      isStreaming: true,
    };
    appendMessages(userMsg, aiMsg);
    // 会话标题取首问
    if (activeSession && activeSession.title === '新对话') {
      setSessions(prev => prev.map(s => (s.id === activeSession.id ? { ...s, title: question.slice(0, 18) } : s)));
    }
    setInput('');
    setIsLoading(true);
    setStage(null);

    // Agent 模式：SSE 流式（支持中途停止）（支持中途停止）
    const controller = new AbortController();
    abortRef.current = controller;
    try {
      const response = await fetch(`${API_URL}/chat/stream`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question, mode, user_id: activeSession?.userId || 'default' }),
        signal: controller.signal,
      });
      if (!response.ok) {
        let detail = `HTTP ${response.status}`;
        try { const d = await response.json(); detail = d.detail || detail; } catch { /* 忽略 */ }
        throw new Error(detail);
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let fullContent = '';
      let image = [];
      let references = [];
      let chartJson = null;
      let lineBuffer = ''; // SSE 事件可能跨 TCP 分片，必须缓存残行
      let sawReports = false; // 本次是否走过研报检索（用于引用缺失自诊断）
      let finalPhase = false; // final 之后的终稿阶段：静默累积，等 done 一次性落地

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        lineBuffer += decoder.decode(value, { stream: true });
        const lines = lineBuffer.split('\n');
        lineBuffer = lines.pop() || ''; // 最后一段可能不完整，留到下个分片再拼

        for (const line of lines) {
          if (!line.startsWith('data: ')) continue;
          const data = line.slice(6);
          if (data === '[DONE]') continue;

          try {
            const parsed = JSON.parse(data);
            if (parsed.type === 'meta') {
              image = parsed.image || [];
              references = parsed.references || [];
              if (parsed.chart_json) chartJson = parsed.chart_json;
            } else if (parsed.type === 'stage') {
              setStage(parsed.stage);
              if (parsed.stage === 'search_reports') sawReports = true;
            } else if (parsed.type === 'final') {
              // 混合题：草稿与终稿高度重复，逐字重放会像"打印两遍"——改为静默累积，等 done 一次性落地
              finalPhase = true;
              fullContent = '';
            } else if (parsed.type === 'content') {
              fullContent += parsed.text;
              if (!finalPhase) { // 终稿阶段不逐字重放
                updateMessage(aiMsgId, { content: fullContent, image, references, chart_json: chartJson });
              }
            } else if (parsed.type === 'done') {
              setStage(null);
              updateMessage(aiMsgId, { content: fullContent, image, references, chart_json: chartJson, isStreaming: false, refsMissing: sawReports && references.length === 0 });
            } else if (parsed.type === 'error') {
              setStage(null);
              updateMessage(aiMsgId, { content: `❌ ${parsed.message || '生成出错'}`, isStreaming: false, error: true });
            }
          } catch (e) {
            // 忽略解析错误
          }
        }
      }
    } catch (error) {
      if (error.name === 'AbortError') {
        // 用户点击「停止」中断
        updateMessage(aiMsgId, { content: '⏹ 已停止生成', isStreaming: false, stopped: true });
      } else {
        console.error('请求失败:', error);
        updateMessage(aiMsgId, { content: '❌ 请求失败，请检查后端服务是否启动', isStreaming: false, error: true, question });
      }
    } finally {
      abortRef.current = null;
      setIsLoading(false);
      setStage(null);
    }
  };

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  };

  const clearHistory = () => {
    if (!confirmClear) {
      setConfirmClear(true);
      setTimeout(() => setConfirmClear(false), 2500);
      return;
    }
    patchActiveMessages(() => []);
    setConfirmClear(false);
  };

  // 渲染图片（点击放大）
  const renderImages = (msg) => {
    if (!msg.image || msg.image.length === 0) return null;
    return (
      <div className="mt-3 grid gap-2">
        {msg.image.map((img, idx) => (
          <img
            key={idx}
            src={resolveImageUrl(img)}
            alt="图表"
            className="max-w-full rounded-lg border border-gray-200 cursor-zoom-in hover:opacity-90 transition-opacity"
            onClick={() => setPreviewImage(resolveImageUrl(img))}
          />
        ))}
      </div>
    );
  };

  // 渲染引用（可展开原文）
  const renderReferences = (msg) => {
    if (!msg.references || msg.references.length === 0) return null;
    // 演示模式只展示第一条引用的完整格式框架，其余折成省略号（保留真实条数）
    const refs = DEMO_MASK ? msg.references.slice(0, 1) : msg.references;
    const hiddenRefs = msg.references.length - refs.length;

    // 高亮命中数字在原文上下文中的位置（展示前清洗 LaTeX 排版残留）
    const renderHitContext = (h) => {
      const context = deLatex(h.context);
      const idx = context.indexOf(h.num);
      if (idx < 0) return context;
      return (
        <>
          {context.slice(0, idx)}
          <mark className="bg-yellow-100 text-green-700 font-semibold px-0.5 rounded">{h.num}</mark>
          {context.slice(idx + h.num.length)}
        </>
      );
    };
    return (
      <div className="mt-3 space-y-1.5">
        <div className="font-medium text-gray-600 text-sm">📚 参考来源（{msg.references.length}）</div>
        {refs.map((ref, idx) => {
          const citation = ref.citation;
          const unhit = citation?.unhit || [];
          return (
            <div key={idx} className="border border-gray-200 rounded-lg p-2 bg-gray-50/60">
              <div className="flex items-center justify-between gap-2">
                <span className="text-gray-700 text-xs font-medium truncate" title={DEMO_MASK ? '' : (citation?.located || ref.paper_path || '')}>
                  {idx + 1}. {mask(basename(ref.paper_path) || '未知来源', MASK_REF_NAME)}
                </span>
                <CitationBadge citation={citation} />
              </div>
              {ref.paper_image && (
                <div className="text-[11px] text-gray-400 truncate mt-1">📊 {mask(basename(ref.paper_image), MASK_REF_IMAGE)}</div>
              )}
              {citation && citation.nums > 0 && (
                citation.num_hit === citation.nums ? (
                  <div className="text-[11px] text-gray-400 mt-1">✓ {citation.nums} 个数字已与研报原文核对</div>
                ) : (
                  <details className="mt-1" open>
                    <summary className="text-[11px] text-orange-600 cursor-pointer select-none">
                      数字核验明细（{unhit.length} 个未命中）
                    </summary>
                    <div className="text-[11px] text-gray-500 mt-1 space-y-0.5 bg-white/70 rounded p-1.5">
                      <div>
                        数字命中：
                        <span className="text-orange-600 font-medium">{citation.num_hit}/{citation.nums}</span>
                      </div>
                      <div>未命中数字：<span className="text-red-600">{unhit.join('、')}</span></div>
                    </div>
                  </details>
                )
              )}
              {ref.text && (
                <details className="mt-1">
                  <summary className="text-[11px] text-gray-400 cursor-pointer hover:text-gray-600 select-none">
                    {`原文片段（${ref.chunk_total ? `第 ${ref.chunk_index + 1}/${ref.chunk_total} 块 · ` : ''}${ref.text_truncated ? `前 ${ref.text.length} 字 / 共 ${ref.text_total} 字` : `共 ${ref.text_total || ref.text.length} 字`}）`}
                  </summary>
                  <div className="text-[11px] text-gray-500 mt-1 whitespace-pre-wrap max-h-28 overflow-y-auto">{mask(ref.text, MASK_REF_TEXT)}</div>
                </details>
              )}

              {citation && citation.hits_context && citation.hits_context.length > 0 && (
                <details className="mt-1">
                  <summary className="text-[11px] text-gray-400 cursor-pointer hover:text-gray-600 select-none">
                    数字命中位置（{citation.hits_context.length} 处 · 研报全文上下文）
                  </summary>
                  <div className="text-[11px] text-gray-500 mt-1 space-y-1 bg-white/70 rounded p-1.5 max-h-40 overflow-y-auto">
                    {citation.hits_context.map((h, i) => (
                      <div key={i} className="leading-relaxed">
                        <span className="text-green-700 font-medium mr-1">#{h.num}</span>
                        <span>…{DEMO_MASK ? MASK_REF_TEXT : renderHitContext(h)}…</span>
                      </div>
                    ))}
                  </div>
                  {!DEMO_MASK && (
                    <div className="text-[10px] text-gray-400 mt-1 pt-1 border-t border-gray-200/70">
                      原文含 LaTeX 排版残留，展示前已清洗
                    </div>
                  )}
                </details>
              )}
            </div>
          );
        })}
        {hiddenRefs > 0 && (
          <div className="text-center text-xs text-gray-400 py-1 select-none">
            ⋯ 其余 {hiddenRefs} 条已省略 ⋯
          </div>
        )}
      </div>
    );
  };

  // 渲染单条消息内容
  const renderMessage = (msg) => {
    if (msg.role === 'user') {
      return <div className="whitespace-pre-wrap break-words">{msg.content}</div>;
    }

    return (
      <div className="space-y-2">
        {DEMO_MASK && !msg.error && !msg.stopped ? (
          <DemoAnswer />
        ) : (
          <div className="prose prose-sm max-w-none">
            <Markdown
              remarkPlugins={[remarkGfm]}
              rehypePlugins={[rehypeHighlight]}
            >
              {msg.content}
            </Markdown>
          </div>
        )}

        {renderImages(msg)}
        {msg.chart_json ? <ChartView option={msg.chart_json} /> : null}
        {renderReferences(msg)}
        {msg.refsMissing && (
          <div className="mt-2 text-xs text-amber-700 bg-amber-50 border border-amber-200 rounded-lg px-2 py-1">
            ⚠️ 本次未收到引用数据（流式事件可能被截断），刷新页面重试即可
          </div>
        )}

        {msg.isStreaming && !msg.content && stage && (
          <div className="flex items-center gap-2 mt-1 text-sm text-gray-500">
            <span className="inline-block w-4 h-4 border-2 border-blue-500 border-t-transparent rounded-full animate-spin" />
            <span>{STAGE_COPY[stage] || '正在处理…'}</span>
          </div>
        )}

        {msg.isStreaming && msg.content && (
          <span className="inline-flex items-center gap-1 mt-1">
            <span className="typing-dot bg-blue-500" />
            <span className="typing-dot bg-blue-500" style={{ animationDelay: '0.15s' }} />
            <span className="typing-dot bg-blue-500" style={{ animationDelay: '0.3s' }} />
          </span>
        )}
      </div>
    );
  };

  const currentMode = MODES.find(m => m.key === mode) || MODES[0];
  const healthDot = health.status === 'ok' ? 'bg-green-500' : health.status === 'down' ? 'bg-red-500' : 'bg-gray-400 animate-pulse';

  return (
    <div className="app-bg flex h-screen">
      {/* 会话侧边栏（最多 5 个，独立 user_id 隔离后端记忆） */}
      <aside className="w-60 shrink-0 border-r border-gray-200/70 bg-white/70 backdrop-blur flex flex-col">
        <div className="p-3 border-b border-gray-200/60 flex items-center justify-between gap-2">
          <span className="text-sm font-semibold text-gray-700">会话</span>
          <button
            onClick={newSession}
            disabled={isLoading}
            className="px-2 py-1 rounded-lg text-xs bg-blue-500 hover:bg-blue-600 text-white shadow-sm disabled:opacity-50"
            title="新建会话（最多 5 个）"
          >
            ＋ 新对话
          </button>
        </div>
        <div className="flex-1 overflow-y-auto p-2 space-y-1">
          {sessions.map(s => (
            <div
              key={s.id}
              onClick={() => { setActiveId(s.id); setStage(null); }}
              className={`group flex items-center gap-1 rounded-lg px-2 py-1.5 cursor-pointer ${s.id === (activeSession?.id) ? 'bg-blue-50 text-blue-700' : 'text-gray-600 hover:bg-gray-100'} ${isLoading ? 'pointer-events-none opacity-60' : ''}`}
              title={`${s.messages.length} 条消息`}
            >
              <span className="flex-1 truncate text-xs">💬 {s.title}</span>
              <button
                onClick={(e) => { e.stopPropagation(); deleteSession(s.id); }}
                className="opacity-0 group-hover:opacity-100 text-gray-400 hover:text-red-500 text-xs px-1"
                title="删除会话"
              >
                ✕
              </button>
            </div>
          ))}
        </div>
        <div className="p-2 text-[10px] text-gray-400 border-t border-gray-200/60">
          {sessions.length}/{MAX_SESSIONS} 个会话
          {sessionHint && <span className="text-orange-500"> · 最多 {MAX_SESSIONS} 个</span>}
        </div>
      </aside>

      <div className="flex flex-col flex-1 min-w-0 max-w-4xl mx-auto w-full px-4">
        {/* 头部 */}
        <header className="flex items-center justify-between py-3 border-b border-gray-200/70 gap-3 shrink-0">
          <div className="flex items-center gap-3 min-w-0">
            <div className="w-9 h-9 rounded-xl bg-gradient-to-br from-blue-500 to-indigo-500 flex items-center justify-center text-lg shadow-sm shrink-0">📊</div>
            <div className="min-w-0">
              <h1 className="text-lg font-bold text-gray-800 leading-tight">智能问数助手</h1>
              <div className="flex items-center gap-1.5 text-xs text-gray-500" title={health.collection ? `向量集合: ${health.collection}` : ''}>
                <span className={`inline-block w-1.5 h-1.5 rounded-full ${healthDot}`} />
                {health.status === 'ok' ? '后端在线' : health.status === 'down' ? '后端离线' : '连接中…'}
              </div>
            </div>
          </div>
          <div className="flex items-center gap-2">
            {DEMO_MASK && (
              <span className="px-2 py-0.5 rounded-md text-xs border border-amber-300 bg-amber-50 text-amber-700 whitespace-nowrap" title="演示模式：答案与引用内容已按数据版权脱敏，界面结构保持真实">
                🔒 演示模式 · 内容已脱敏
              </span>
            )}
            <button
              onClick={clearHistory}
              className={`px-2.5 py-1.5 rounded-lg text-xs border transition-colors ${
                confirmClear ? 'bg-red-500 text-white border-red-500' : 'text-gray-500 border-gray-300 hover:bg-gray-100'
              }`}
              title="清空当前会话"
            >
              {confirmClear ? '确认清空？' : '🗑 清空'}
            </button>
          </div>
        </header>

        {/* 模式说明条 */}
        <div className="flex items-center justify-between pt-2 text-xs text-gray-400 shrink-0">
          <span className={`px-2 py-0.5 rounded-md ${currentMode.badge}`}>{currentMode.label}：{currentMode.desc}</span>
          {messages.length > 0 && <span>共 {messages.length} 条消息</span>}
        </div>
        <div className="text-[11px] text-gray-400 pt-1 shrink-0">💡 Agent 分析：既能查研报观点，也能查财务数字并出图表。</div>

        {/* 消息列表 */}
        <div className="flex-1 overflow-y-auto py-4 space-y-4">
          {messages.length === 0 && (
            <div className="flex flex-col items-center justify-center h-full text-center px-4">
              <div className="text-6xl mb-4">💬</div>
              <p className="text-gray-500 text-base">你好，我是智能问数助手</p>
              <p className="text-sm text-gray-400 mt-1 mb-6">{currentMode.label}模式 · {currentMode.desc}</p>
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-2 w-full max-w-lg">
                {(SUGGESTIONS[mode] || []).map((s, idx) => (
                  <button
                    key={idx}
                    onClick={() => sendMessage(s.q)}
                    disabled={isLoading}
                    className="text-left px-3 py-2.5 rounded-xl border border-gray-200 bg-white/80 hover:border-blue-300 hover:shadow-sm transition-all text-sm text-gray-600 flex items-center gap-2 disabled:opacity-50"
                  >
                    <span>{s.icon}</span>
                    <span className="truncate">{s.q}</span>
                  </button>
                ))}
              </div>
            </div>
          )}

          {messages.map((msg, idx) => (
            <div key={msg.id || idx} className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}>
              <div className={`flex gap-2 max-w-[82%] ${msg.role === 'user' ? 'flex-row-reverse' : ''}`}>
                {/* 头像 */}
                <div className={`w-8 h-8 rounded-full flex items-center justify-center text-sm shrink-0 mt-1 ${
                  msg.role === 'user' ? 'bg-blue-100' : 'bg-indigo-100'
                }`}>
                  {msg.role === 'user' ? '🧑‍💻' : '🤖'}
                </div>
                <div className="min-w-0">
                  <div className={`rounded-2xl px-4 py-3 ${
                    msg.role === 'user'
                      ? 'bg-blue-500 text-white'
                      : 'bg-white border border-gray-200 shadow-sm text-gray-800'
                  }`}>
                    {renderMessage(msg)}
                  </div>

                  {/* 消息操作栏 */}
                  {msg.role === 'user' ? (
                    <div className="flex justify-end mt-1 pr-1">
                      <span className="text-[10px] text-gray-400">{formatTime(msg.ts)}</span>
                    </div>
                  ) : (
                    !msg.isStreaming && (
                      <div className="flex items-center gap-2 mt-1 px-1">
                        <span className="text-[10px] text-gray-400">{formatTime(msg.ts)}</span>
                        <button
                          onClick={() => copyText(msg.id, msg.content)}
                          className="text-[11px] text-gray-400 hover:text-gray-600"
                          title="复制回答"
                        >
                          {copiedId === msg.id ? '✓ 已复制' : '📋 复制'}
                        </button>
                        {msg.error && msg.question && (
                          <button
                            onClick={() => sendMessage(msg.question)}
                            className="text-[11px] text-blue-500 hover:text-blue-600"
                          >
                            🔄 重试
                          </button>
                        )}
                      </div>
                    )
                  )}
                </div>
              </div>
            </div>
          ))}
          <div ref={messagesEndRef} />
        </div>

        {/* 输入区 */}
        <div className="border-t border-gray-200/70 pt-3 pb-3 shrink-0">
          <div className="flex gap-3 items-end">
            <textarea
              ref={taRef}
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder={currentMode.placeholder || '输入问题...（Enter发送，Shift+Enter换行）'}
              className="flex-1 resize-none rounded-xl border border-gray-300 px-4 py-2.5 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent text-sm"
              rows={1}
            />
            {isLoading ? (
              <button
                onClick={stopGeneration}
                className="px-6 py-2.5 rounded-xl font-medium bg-red-500 hover:bg-red-600 text-white text-sm shrink-0"
              >
                ⏹ 停止
              </button>
            ) : (
              <button
                onClick={() => sendMessage()}
                disabled={!input.trim()}
                className={`px-6 py-2.5 rounded-xl font-medium text-sm shrink-0 transition-colors ${
                  !input.trim()
                    ? 'bg-gray-200 text-gray-400 cursor-not-allowed'
                    : 'bg-blue-500 hover:bg-blue-600 text-white shadow-sm'
                }`}
              >
                发送
              </button>
            )}
          </div>
          <div className="text-[11px] text-gray-400 mt-1.5 px-1">
            {isLoading
              ? ((stage ? (STAGE_COPY[stage] || '正在处理…') : '正在生成') + '，点击「停止」可中断…')
              : (currentMode.label + '：' + currentMode.desc)}
          </div>
        </div>
      </div>

      {/* 图片预览弹窗 */}
      {previewImage && (
        <div
          className="fixed inset-0 bg-black/70 z-50 flex items-center justify-center p-4"
          onClick={() => setPreviewImage(null)}
        >
          <img src={previewImage} alt="预览" className="max-w-full max-h-full rounded-lg shadow-2xl" />
          <button
            className="absolute top-4 right-4 w-9 h-9 rounded-full bg-white/20 hover:bg-white/40 text-white text-lg"
            onClick={() => setPreviewImage(null)}
          >
            ✕
          </button>
        </div>
      )}
    </div>
  );
}

export default App;
