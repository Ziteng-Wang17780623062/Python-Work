// 通用JavaScript函数
const API_BASE = window.location.origin;

function initModule() {
    const form = document.getElementById('evaluation-form');
    const submitBtn = document.getElementById('submit-btn');
    const MODULE_NAME = window.MODULE_NAME;

    form.addEventListener('submit', async (e) => {
        e.preventDefault();
        const formData = new FormData(form);
        const params = {};

        for (const [key, value] of formData.entries()) {
            if (value === '') continue;
            if (key === 'limit' || key === 'sample_size' || key === 'seed') {
                params[key] = parseInt(value) || null;
            } else {
                params[key] = value;
            }
        }

        submitBtn.disabled = true;
        submitBtn.textContent = '运行中...';

        try {
            const response = await fetch(`${API_BASE}/api/run`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ module: MODULE_NAME, params: params })
            });

            const data = await response.json();
            if (data.error) throw new Error(data.error);

            showResultPanel();
            pollTaskStatus(data.task_id, submitBtn);
        } catch (error) {
            showError(error.message);
            submitBtn.disabled = false;
            submitBtn.textContent = '开始测评';
        }
    });
}

function showResultPanel() {
    document.getElementById('result-panel').classList.add('show');
    document.getElementById('status-container').innerHTML = '';
    document.getElementById('output-container').innerHTML = '';
}

function updateStatus(status, message) {
    document.getElementById('status-container').innerHTML = `
        <div class="status ${status}">
            ${status === 'running' ? '<span class="loading"></span>' : ''}
            ${message}
        </div>
    `;
}

function filterOutput(text) {
    if (!text) return '';
    
    // 按行分割
    const lines = text.split('\n');
    const filteredLines = [];
    
    for (const line of lines) {
        // 跳过空行
        if (line.trim() === '') {
            continue;
        }
        
        // 跳过进度条（包含 %| 或类似格式）
        // 匹配格式：xxx: 100%|#####| 1/1 [00:02<00:00,  2.24s/样本]
        // 或：xxx:   0%|             | 0/1 [00:00<?, ?样本/s]
        if (/%\s*\||\|\s*\d+\/\d+\s*\[/.test(line)) {
            continue;
        }
        
        // 跳过只包含空格和特殊字符的行
        if (/^[\s\u00A0\u2000-\u200B\u2028\u2029\u3000]*$/.test(line)) {
            continue;
        }
        
        // 保留其他行
        filteredLines.push(line);
    }
    
    return filteredLines.join('\n');
}

function updateOutput(output, isError = false) {
    const container = document.getElementById('output-container');
    // 过滤输出，移除进度条和空行（错误输出不过滤）
    const filteredOutput = isError ? output : filterOutput(output);
    container.innerHTML = `<div class="output ${isError ? 'error' : 'success'}">${escapeHtml(filteredOutput)}</div>`;
}

function showError(message) {
    updateStatus('failed', `错误: ${message}`);
    updateOutput(message, true);
}

async function pollTaskStatus(taskId, submitBtn) {
    const maxAttempts = 3600;
    let attempts = 0;

    const interval = setInterval(async () => {
        attempts++;
        if (attempts > maxAttempts) {
            clearInterval(interval);
            updateStatus('failed', '任务超时');
            submitBtn.disabled = false;
            submitBtn.textContent = '开始测评';
            return;
        }

        try {
            const response = await fetch(`${API_BASE}/api/status/${taskId}`);
            const data = await response.json();

            if (data.status === 'running') {
                updateStatus('running', '任务运行中，请稍候...');
            } else if (data.status === 'completed') {
                clearInterval(interval);
                updateStatus('completed', '任务完成！');
                let output = '';
                if (data.result?.stdout) output += data.result.stdout;
                if (data.result?.stderr) {
                    // 过滤stderr中的进度条和空行
                    const filteredStderr = filterOutput(data.result.stderr);
                    if (filteredStderr.trim()) {
                        output += '\n\n错误输出:\n' + filteredStderr;
                    }
                }
                updateOutput(output || '任务执行成功，但无输出内容');
                submitBtn.disabled = false;
                submitBtn.textContent = '开始测评';
            } else if (data.status === 'failed') {
                clearInterval(interval);
                updateStatus('failed', '任务失败');
                let errorMsg = data.error || '未知错误';
                if (data.result?.stderr) errorMsg += '\n\n' + data.result.stderr;
                if (data.result?.stdout) errorMsg += '\n\n标准输出:\n' + data.result.stdout;
                updateOutput(errorMsg, true);
                submitBtn.disabled = false;
                submitBtn.textContent = '开始测评';
            }
        } catch (error) {
            clearInterval(interval);
            showError('无法获取任务状态: ' + error.message);
            submitBtn.disabled = false;
            submitBtn.textContent = '开始测评';
        }
    }, 1000);
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

