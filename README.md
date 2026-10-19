<div align="center">

# 🏥 SIAT-BR · MPAT

**Sistema Integrado de Assistência e Telefarmácia Nacional**
*MPAT: Motor de Predição de Adesão Terapêutica*

Protótipo funcional desenvolvido para a participação no **32º Prêmio Jovem Cientista (CNPq, 2026)**
Tema: *Inteligência Artificial para o Bem Comum*

</div>

> ⚠️ **Aviso importante.** Este é um **protótipo educacional e de pesquisa**. **Todos os dados são simulados** (pacientes, documentos e números fictícios). Não é dispositivo médico, não faz diagnóstico, não prescreve e **não substitui o acompanhamento de profissionais de saúde habilitados**. As mensagens, o chat e as orientações do app são ilustrativos. Os indicadores exibidos no painel vêm de dados simulados e não representam resultados reais.

## 💡 A ideia

Muitos tratamentos de longa duração, como os de agravos negligenciados, falham porque a pessoa **deixa de retirar ou de tomar a medicação**. O SIAT-BR imagina uma telefarmácia que ajuda a **perceber cedo quem precisa de apoio** e a **conversar com o cidadão em linguagem simples**, apoiando (nunca substituindo) a decisão do profissional.

## 🧭 O que o protótipo mostra

| Aba | O que você encontra |
|---|---|
| 👤 **Cidadão** | Farol de adesão (verde, amarelo, vermelho), guia do tratamento com validação diária da dose, linguagem em níveis progressivos (incluindo um jogo, a **Batalha Terapêutica**), registro de sintomas e chat guia |
| 🩺 **Cockpit do farmacêutico** | Formulário de triagem, **score de risco de abandono** com justificativa (IA explicável) e condutas sugeridas como protocolos de exemplo |
| 📊 **Dashboard** | Panorama simulado de Salvador/BA: risco por bairro, importância das variáveis, distribuição de agravos e relação entre cobertura e vulnerabilidade |
| 📜 **Governança** | Pseudonimização, princípios da LGPD, soberania do julgamento profissional e declaração de uso de IA |

<details>
<summary>🔎 <strong>Como funciona por dentro</strong> (clique para abrir)</summary>

- **Dados:** gerados de forma sintética para uma amostra de pacientes fictícios de Salvador/BA, com agravos como tuberculose, hanseníase, leishmaniose visceral e doença de Chagas.
- **Modelo:** classificador de floresta aleatória (*Random Forest*) que estima o risco de abandono a partir de cobertura da medicação, efeitos adversos relatados, vulnerabilidade social, polifarmácia, suporte familiar e idade.
- **Explicabilidade:** o app mostra quais fatores pesaram em cada estimativa.
- **Privacidade (simulada):** identificadores fictícios, exibidos de forma pseudonimizada.
- **Limites:** como os dados e o alvo são simulados, as métricas do modelo servem para demonstrar o fluxo, **não** como evidência clínica.

</details>

## ▶️ Como executar

Requer Python 3.10+.

```bash
pip install -r requirements.txt
streamlit run siat_br_app.py
```

O app abre no navegador, em `http://localhost:8501`.

## 🌱 Possibilidades de evolução

Ideias em avaliação, sem prazo definido: validação com especialistas e serviços de saúde, integração com sistemas reais com as devidas autorizações e proteções, revisão ética e jurídica, e testes de usabilidade com a população atendida.

## 👤 Autoria

Criado por **Daniel Pires Francisco** (Daniel P Francisco), Presidente da LAIFT e graduando em Farmácia na UNINASSAU Salvador.

- Portfólio, currículo e serviços: https://dpflabs.com
- Plataforma LAIFT: https://laift.com.br
- Lattes: http://lattes.cnpq.br/5041510175000740
